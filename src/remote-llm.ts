/**
 * remote-llm.ts - Remote LLM provider for OpenAI-compatible embedding & reranking servers.
 *
 * Supports:
 * - POST /v1/embeddings (OpenAI-compatible)
 * - POST /v1/rerank (Cohere-compatible)
 *
 * Used with servers like omlx that serve MLX format models (e.g. bge-m3, bge-reranker).
 */

import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  RerankOptions,
  RerankResult,
  RerankDocument,
  Queryable,
} from "./llm.js";

export type RemoteLLMConfig = {
  /** Base URL for the API (e.g. "http://localhost:8000/v1") */
  baseUrl: string;
  /** Optional API key for authentication */
  apiKey?: string;
  /** Embedding model name (e.g. "bge-m3") */
  embedModel?: string;
  /** Reranking model name (e.g. "bge-reranker-v2-m3") */
  rerankModel?: string;
  /** Request timeout in ms for embeddings (default: 30000) */
  timeoutMs?: number;
  /** Request timeout in ms for reranking (default: 300000 = 5 min, reranking is slower) */
  rerankTimeoutMs?: number;
};

const debug = !!process.env.QMD_REMOTE_DEBUG;
const INDIVIDUAL_RETRY_DELAY_MS = 100;
const MAX_INDIVIDUAL_EMBED_RETRIES = 10;
const RERANK_BATCH_SIZE = 5;
const RERANK_MAX_CHARS_PER_DOC = 256;
const MAX_RERANK_RETRIES = 3;
const INITIAL_RERANK_RETRY_DELAY_MS = 500;

function normalizeRemoteBaseUrl(baseUrl: string): string {
  const normalized = baseUrl.replace(/\/+$/, "");
  if (!/^https?:\/\//.test(normalized)) {
    throw new Error("Remote LLM baseUrl must start with http:// or https://");
  }
  return normalized;
}

function truncateRemoteErrorBody(body: string): string {
  return body.slice(0, 200);
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function sanitizeRerankText(text: string): string {
  return text
    .replace(
      /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]|\uFFFD/g,
      "",
    )
    .replace(/\r\n/g, " ")
    .replace(/\n/g, " ")
    .replace(/\t/g, " ")
    .replace(/[`~]/g, "")
    .replace(/[\u200B-\u200D\uFEFF]/g, "")
    .replace(/[\x00-\x1f\x7f-\x9f]/g, "")
    .replace(/ {2,}/g, " ")
    .trim();
}

function truncateRerankDocumentText(text: string, index: number): string {
  const clean = sanitizeRerankText(text);

  if (clean.length <= RERANK_MAX_CHARS_PER_DOC) {
    return clean;
  }

  const truncatedAt = clean.lastIndexOf(" ", RERANK_MAX_CHARS_PER_DOC);
  const cleanLimit = truncatedAt > 0 ? truncatedAt : RERANK_MAX_CHARS_PER_DOC;
  const truncated = `${clean.slice(0, cleanLimit)}...`;

  if (debug) {
    process.stderr.write(
      `[remote-llm] rerank: truncated doc[${index}] from ${text.length} to ${truncated.length} chars\n`,
    );
  }

  return truncated;
}

export class RemoteLLM implements LLM {
  private baseUrl: string;
  private apiKey?: string;
  private embedModel: string;
  private rerankModel: string;
  private timeoutMs: number;
  private rerankTimeoutMs: number;

  readonly isRemote = true;

  constructor(config: RemoteLLMConfig) {
    this.baseUrl = normalizeRemoteBaseUrl(config.baseUrl);
    this.apiKey = config.apiKey;
    this.embedModel = config.embedModel ?? "bge-m3";
    this.rerankModel = config.rerankModel ?? "bge-reranker-v2-m3";
    this.timeoutMs = config.timeoutMs ?? 30_000;
    this.rerankTimeoutMs = config.rerankTimeoutMs ?? 300_000;
    if (debug) {
      process.stderr.write(`[remote-llm] init baseUrl=${this.baseUrl} embed=${this.embedModel} rerank=${this.rerankModel}\n`);
    }
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) {
      h["Authorization"] = `Bearer ${this.apiKey}`;
    }
    return h;
  }

  private isFireworksBaseUrl(): boolean {
    try {
      return new URL(this.baseUrl).hostname.endsWith("fireworks.ai");
    } catch {
      return false;
    }
  }

  private normalizeFireworksModel(model: string): string {
    const trimmed = model.trim();
    if (trimmed.startsWith("accounts/fireworks/models/")) {
      return trimmed;
    }
    if (trimmed.startsWith("fireworks/")) {
      return `accounts/fireworks/models/${trimmed.slice("fireworks/".length)}`;
    }
    return trimmed;
  }

  private resolveRemoteRerankModel(requestedModel?: string): string {
    const configuredModel = this.isFireworksBaseUrl()
      ? this.normalizeFireworksModel(this.rerankModel)
      : this.rerankModel;
    const override = requestedModel?.trim();

    if (!override) {
      return configuredModel;
    }

    if (!this.isFireworksBaseUrl()) {
      return override;
    }

    if (
      override.startsWith("fireworks/") ||
      override.startsWith("accounts/fireworks/models/")
    ) {
      return this.normalizeFireworksModel(override);
    }

    if (debug) {
      process.stderr.write(
        `[remote-llm] rerank: ignoring non-Fireworks model override "${override}", using "${configuredModel}"\n`,
      );
    }
    return configuredModel;
  }

  private async fetchWithTimeout(url: string, init: RequestInit, timeoutMs?: number): Promise<Response> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs ?? this.timeoutMs);
    try {
      const resp = await fetch(url, { ...init, redirect: "error", signal: controller.signal });
      if (!resp.ok) {
        const body = await resp.text().catch(() => "");
        throw new Error(`Remote LLM error ${resp.status}: ${truncateRemoteErrorBody(body)}`);
      }
      return resp;
    } finally {
      clearTimeout(timeout);
    }
  }

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    const results = await this.embedBatch([text], options);
    return results[0] ?? null;
  }

  async embedBatch(texts: string[], options?: EmbedOptions): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];
    const model = options?.model ?? this.embedModel;

    // Sanitize inputs to avoid remote tokenizer errors:
    // 1. Replace empty/non-string entries with a space
    // 2. Fix broken Unicode from chunk splitting (unpaired surrogates from emoji split mid-codepoint)
    const sanitized = texts.map((t, i) => {
      if (typeof t !== "string" || t.trim().length === 0) {
        if (debug) process.stderr.write(`[remote-llm] warning: empty text at index ${i}, replacing with placeholder\n`);
        return " ";
      }
      // Remove unpaired surrogates and replacement characters that crash remote tokenizers.
      // This happens when chunking splits a surrogate pair (emoji) in half.
      return t.replace(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]|\uFFFD/g, "");
    });

    if (debug) {
      process.stderr.write(`[remote-llm] POST ${this.baseUrl}/embeddings model=${model} texts=${sanitized.length}\n`);
    }
    const start = Date.now();

    try {
      const resp = await this.fetchWithTimeout(`${this.baseUrl}/embeddings`, {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify({
          model,
          input: sanitized,
        }),
      });

      const json = await resp.json() as {
        data: { embedding: number[]; index: number }[];
        model: string;
      };

      if (debug) {
        const dim = json.data[0]?.embedding.length ?? 0;
        process.stderr.write(`[remote-llm] embed done ${Date.now() - start}ms results=${json.data.length} dim=${dim}\n`);
      }

      // Map response back to input order
      const resultMap = new Map<number, number[]>();
      for (const item of json.data) {
        resultMap.set(item.index, item.embedding);
      }

      return texts.map((_, i) => {
        const embedding = resultMap.get(i);
        if (!embedding) return null;
        return { embedding, model: json.model || model };
      });
    } catch (error) {
      if (debug) {
        const lengths = sanitized.map((t, i) => `[${i}]:${t.length}`).join(" ");
        process.stderr.write(`[remote-llm] embed FAILED batch of ${sanitized.length}, retrying individually. lengths: ${lengths}\n`);
      }
      // Batch failed — retry each text individually to isolate bad inputs
      const results: (EmbeddingResult | null)[] = [];
      for (let i = 0; i < sanitized.length; i++) {
        if (i >= MAX_INDIVIDUAL_EMBED_RETRIES) {
          if (i === MAX_INDIVIDUAL_EMBED_RETRIES) {
            console.warn(
              `[remote-llm] individual embed retry cap hit at ${MAX_INDIVIDUAL_EMBED_RETRIES}; skipping remaining ${sanitized.length - i} items`,
            );
          }
          results.push(null);
          continue;
        }

        if (i > 0) {
          await delay(INDIVIDUAL_RETRY_DELAY_MS);
        }

        try {
          const resp = await this.fetchWithTimeout(`${this.baseUrl}/embeddings`, {
            method: "POST",
            headers: this.headers(),
            body: JSON.stringify({ model, input: [sanitized[i]] }),
          });
          const json = await resp.json() as { data: { embedding: number[]; index: number }[]; model: string };
          const emb = json.data[0]?.embedding;
          results.push(emb ? { embedding: emb, model: json.model || model } : null);
        } catch (e) {
          if (debug) {
            const preview = sanitized[i]!.slice(0, 120).replace(/\n/g, "\\n");
            process.stderr.write(`[remote-llm] embed single[${i}] FAILED len=${sanitized[i]!.length} preview="${preview}": ${e}\n`);
          }
          results.push(null);
        }
      }
      return results;
    }
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions
  ): Promise<RerankResult> {
    const model = this.resolveRemoteRerankModel(options?.model);
    if (documents.length === 0) {
      return { results: [], model };
    }

    const texts = documents.map((d, index) => truncateRerankDocumentText(d.text, index));

    if (debug) {
      process.stderr.write(`[remote-llm] POST ${this.baseUrl}/rerank model=${model} docs=${texts.length} query="${query.slice(0, 60)}"\n`);
    }
    const start = Date.now();

    try {
      const isRetryableRerankError = (error: unknown): boolean => {
        if (!(error instanceof Error)) {
          return false;
        }
        const statusMatch = error.message.match(/Remote LLM error (\d+):/);
        if (!statusMatch) {
          return false;
        }
        const status = Number(statusMatch[1]);
        return status === 429 || status >= 500;
      };

      const rerankBatch = async (
        batchTexts: string[],
        offset: number,
        batchIndex: number,
        batchCount: number,
        attempt = 0,
      ): Promise<{ index: number; relevance_score: number }[]> => {
        const batchLabel = `${batchIndex + 1}/${batchCount}`;
        const requestBody = JSON.stringify({
          model,
          query,
          documents: batchTexts,
          return_documents: false,
        });

        if (debug && batchCount > 1) {
          process.stderr.write(
            `[remote-llm] rerank batch ${batchLabel} (${batchTexts.length} docs, ${Buffer.byteLength(requestBody, "utf8")} bytes, attempt ${attempt + 1})...\n`,
          );
        }

        try {
          const resp = await this.fetchWithTimeout(`${this.baseUrl}/rerank`, {
            method: "POST",
            headers: this.headers(),
            body: requestBody,
          }, this.rerankTimeoutMs);

          const json = await resp.json() as {
            results?: { index: number; relevance_score: number }[];
            data?: { index: number; relevance_score: number }[];
          };

          return (json.results ?? json.data ?? []).map((result) => ({
            index: result.index + offset,
            relevance_score: result.relevance_score,
          }));
        } catch (error) {
          if (!isRetryableRerankError(error)) {
            throw error;
          }

          if (attempt + 1 < MAX_RERANK_RETRIES) {
            const delayMs = INITIAL_RERANK_RETRY_DELAY_MS * 2 ** attempt;
            if (debug) {
              process.stderr.write(
                `[remote-llm] rerank batch ${batchLabel} retrying after ${delayMs}ms: ${error}\n`,
              );
            }
            await delay(delayMs);
            return rerankBatch(batchTexts, offset, batchIndex, batchCount, attempt + 1);
          }

          if (batchTexts.length > 1) {
            const midpoint = Math.ceil(batchTexts.length / 2);
            if (debug) {
              process.stderr.write(
                `[remote-llm] rerank batch ${batchLabel} splitting ${batchTexts.length} docs after retries exhausted\n`,
              );
            }
            const firstHalf = await rerankBatch(
              batchTexts.slice(0, midpoint),
              offset,
              batchIndex,
              batchCount,
            );
            const secondHalf = await rerankBatch(
              batchTexts.slice(midpoint),
              offset + midpoint,
              batchIndex,
              batchCount,
            );
            return [...firstHalf, ...secondHalf];
          }

          throw error;
        }
      };

      let ranked: { index: number; relevance_score: number }[];
      if (documents.length <= RERANK_BATCH_SIZE) {
        ranked = await rerankBatch(texts, 0, 0, 1);
      } else {
        const batchCount = Math.ceil(texts.length / RERANK_BATCH_SIZE);
        ranked = [];

        for (let batchIndex = 0; batchIndex < batchCount; batchIndex++) {
          const offset = batchIndex * RERANK_BATCH_SIZE;
          const batchTexts = texts.slice(offset, offset + RERANK_BATCH_SIZE);
          const batchResults = await rerankBatch(batchTexts, offset, batchIndex, batchCount);
          ranked.push(...batchResults);
        }

        ranked.sort((a, b) => b.relevance_score - a.relevance_score);
      }

      if (debug) {
        const top = ranked[0];
        process.stderr.write(`[remote-llm] rerank done ${Date.now() - start}ms results=${ranked.length} top_score=${top?.relevance_score?.toFixed(4) ?? "N/A"}\n`);
      }

      const results = ranked.map((r) => ({
        file: documents[r.index]?.file ?? "",
        score: r.relevance_score,
        index: r.index,
      }));

      return { results, model };
    } catch (error) {
      console.error("Remote rerank error:", error);
      throw error;
    }
  }

  async generate(_prompt: string, _options?: GenerateOptions): Promise<GenerateResult | null> {
    throw new Error("RemoteLLM does not support generate(). Use local LlamaCpp for query expansion.");
  }

  async expandQuery(_query: string, _options?: { context?: string; includeLexical?: boolean; intent?: string }): Promise<Queryable[]> {
    throw new Error("RemoteLLM does not support expandQuery(). Use local LlamaCpp for query expansion.");
  }

  async modelExists(model: string): Promise<ModelInfo> {
    try {
      const resp = await this.fetchWithTimeout(`${this.baseUrl}/models`, {
        method: "GET",
        headers: this.headers(),
      });
      const json = await resp.json() as { data?: { id: string }[] };
      const models = json.data ?? [];
      const exists = models.some((m) => m.id === model);
      return { name: model, exists };
    } catch {
      return { name: model, exists: false };
    }
  }

  async dispose(): Promise<void> {
    // No-op: no local resources to clean up
  }
}

/**
 * full-remote-llm.ts - Fully remote LLM provider for OpenAI-compatible APIs.
 *
 * Routes all operations remotely:
 * - POST /v1/embeddings
 * - POST /v1/rerank
 * - POST /v1/chat/completions
 */

import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  Queryable,
  QueryType,
  RerankDocument,
  RerankOptions,
  RerankResult,
} from "./llm.js";
import { RemoteLLM, type RemoteLLMConfig } from "./remote-llm.js";

export type FullRemoteLLMConfig = RemoteLLMConfig & {
  /** Chat model used for query expansion */
  expandModel?: string;
  /** Chat model used for text generation */
  generateModel?: string;
};

type ChatCompletionResponse = {
  model?: string;
  choices?: Array<{
    text?: string;
    message?: {
      content?:
        | string
        | Array<
            | string
            | {
                text?: string;
                content?: string;
              }
          >;
    };
  }>;
};

const DEFAULT_CHAT_MODEL = "accounts/fireworks/models/qwen3-8b";
const debug = !!process.env.QMD_REMOTE_DEBUG;
const EXPAND_SYSTEM_PROMPT =
  "You are a search query expansion assistant. Given a search query, generate expanded search terms in exactly this format — one per line, no other output:\n" +
  "lex: keyword1 keyword2 keyword3\n" +
  'lex: "exact phrase match"\n' +
  "vec: semantic meaning of the query expressed differently\n" +
  "vec: related concept that would match relevant documents\n" +
  "hyde: A short paragraph written as if it were part of an ideal matching document\n\n" +
  "Rules:\n" +
  "- Generate 2-3 lex lines (keywords and exact phrases for BM25 search)\n" +
  "- Generate 2-3 vec lines (semantic variations for vector search)\n" +
  "- Generate 1 hyde line (hypothetical document excerpt)\n" +
  "- Every line MUST start with lex:, vec:, or hyde:\n" +
  "- Include at least one term from the original query in each line\n" +
  "- No explanations, no markdown, no extra text";

function sanitizeRemoteText(text: string, fallback = " "): string {
  if (typeof text !== "string" || text.trim().length === 0) {
    return fallback;
  }
  return text.replace(
    /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]|\uFFFD/g,
    "",
  );
}

function extractChatText(json: ChatCompletionResponse): string {
  const choice = json.choices?.[0];
  if (!choice) return "";

  if (typeof choice.text === "string") {
    return choice.text;
  }

  const content = choice.message?.content;
  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        if (typeof part?.text === "string") return part.text;
        if (typeof part?.content === "string") return part.content;
        return "";
      })
      .join("");
  }

  return "";
}

function parseExpandedQuery(
  responseText: string,
  query: string,
  includeLexical: boolean,
): Queryable[] {
  const lines = responseText.trim().split("\n");
  const queryTerms = query
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);

  const hasQueryTerm = (text: string): boolean => {
    const lower = text.toLowerCase();
    if (queryTerms.length === 0) return true;
    return queryTerms.some((term) => lower.includes(term));
  };

  const queryables: Queryable[] = lines
    .map((line) => {
      const colonIdx = line.indexOf(":");
      if (colonIdx === -1) return null;

      const type = line.slice(0, colonIdx).trim();
      if (type !== "lex" && type !== "vec" && type !== "hyde") return null;

      const text = line.slice(colonIdx + 1).trim();
      if (!hasQueryTerm(text)) return null;

      return { type: type as QueryType, text };
    })
    .filter((q): q is Queryable => q !== null);

  const filtered = includeLexical
    ? queryables
    : queryables.filter((q) => q.type !== "lex");

  if (filtered.length > 0) {
    return filtered;
  }

  const fallback: Queryable[] = [
    { type: "hyde", text: `Information about ${query}` },
    { type: "lex", text: query },
    { type: "vec", text: query },
  ];
  return includeLexical ? fallback : fallback.filter((q) => q.type !== "lex");
}

export class FullRemoteLLM implements LLM {
  private remote: RemoteLLM;
  private baseUrl: string;
  private apiKey?: string;
  private timeoutMs: number;
  private expandModel: string;
  private generateModel: string;

  readonly isRemote = true;

  constructor(config: FullRemoteLLMConfig) {
    this.baseUrl = config.baseUrl.replace(/\/+$/, "");
    this.apiKey = config.apiKey;
    this.timeoutMs = config.timeoutMs ?? 30_000;
    this.expandModel = config.expandModel ?? DEFAULT_CHAT_MODEL;
    this.generateModel = config.generateModel ?? this.expandModel;
    this.remote = new RemoteLLM({
      baseUrl: this.baseUrl,
      apiKey: this.apiKey,
      embedModel: config.embedModel,
      rerankModel: config.rerankModel,
      timeoutMs: this.timeoutMs,
      rerankTimeoutMs: config.rerankTimeoutMs,
    });

    if (debug) {
      process.stderr.write(
        `[full-remote-llm] init baseUrl=${this.baseUrl} expand=${this.expandModel} generate=${this.generateModel}\n`,
      );
    }
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) {
      h.Authorization = `Bearer ${this.apiKey}`;
    }
    return h;
  }

  private async fetchWithTimeout(
    url: string,
    init: RequestInit,
    timeoutMs?: number,
  ): Promise<Response> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs ?? this.timeoutMs);
    try {
      const resp = await fetch(url, { ...init, signal: controller.signal });
      if (!resp.ok) {
        const body = await resp.text().catch(() => "");
        throw new Error(`Remote LLM error ${resp.status}: ${body}`);
      }
      return resp;
    } finally {
      clearTimeout(timeout);
    }
  }

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    return this.remote.embed(text, options);
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    return this.remote.embedBatch(texts);
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions,
  ): Promise<RerankResult> {
    return this.remote.rerank(query, documents, options);
  }

  async generate(
    prompt: string,
    options: GenerateOptions = {},
  ): Promise<GenerateResult | null> {
    const model = options.model ?? this.generateModel;
    const sanitizedPrompt = sanitizeRemoteText(prompt);

    if (debug) {
      process.stderr.write(
        `[full-remote-llm] POST ${this.baseUrl}/chat/completions model=${model} prompt_len=${sanitizedPrompt.length}\n`,
      );
    }
    const start = Date.now();

    try {
      const resp = await this.fetchWithTimeout(`${this.baseUrl}/chat/completions`, {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: sanitizedPrompt }],
          max_tokens: options.maxTokens ?? 150,
          temperature: options.temperature ?? 0.7,
        }),
      });

      const json = (await resp.json()) as ChatCompletionResponse;
      const text = extractChatText(json).trim();

      if (debug) {
        process.stderr.write(
          `[full-remote-llm] generate done ${Date.now() - start}ms text_len=${text.length}\n`,
        );
      }

      return {
        text,
        model: json.model || model,
        done: true,
      };
    } catch (error) {
      console.error("Remote generate error:", error);
      return null;
    }
  }

  async expandQuery(
    query: string,
    options: { context?: string; includeLexical?: boolean; intent?: string } = {},
  ): Promise<Queryable[]> {
    const includeLexical = options.includeLexical ?? true;
    const sanitizedQuery = sanitizeRemoteText(query);
    const systemPrompt = options.intent
      ? `${EXPAND_SYSTEM_PROMPT}\nQuery intent: ${sanitizeRemoteText(options.intent)}`
      : EXPAND_SYSTEM_PROMPT;

    if (debug) {
      process.stderr.write(
        `[full-remote-llm] POST ${this.baseUrl}/chat/completions model=${this.expandModel} expand_query_len=${sanitizedQuery.length}\n`,
      );
    }
    const start = Date.now();

    try {
      const messages: Array<{ role: "system" | "user"; content: string }> = [
        { role: "system", content: systemPrompt },
      ];

      if (options.context) {
        messages.push({
          role: "system",
          content: `Query context: ${sanitizeRemoteText(options.context)}`,
        });
      }

      messages.push({ role: "user", content: sanitizedQuery });

      const resp = await this.fetchWithTimeout(`${this.baseUrl}/chat/completions`, {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify({
          model: this.expandModel,
          messages,
          max_tokens: 600,
          temperature: 0.3,
        }),
      });

      const json = (await resp.json()) as ChatCompletionResponse;
      const responseText = extractChatText(json);
      const parsed = parseExpandedQuery(responseText, query, includeLexical);

      if (debug) {
        process.stderr.write(
          `[full-remote-llm] expand done ${Date.now() - start}ms lines=${responseText.trim().split("\n").filter(Boolean).length} parsed=${parsed.length}\n`,
        );
      }

      return parsed;
    } catch (error) {
      console.error("Structured query expansion failed:", error);
      return parseExpandedQuery("", query, includeLexical);
    }
  }

  async modelExists(model: string): Promise<ModelInfo> {
    try {
      const resp = await this.fetchWithTimeout(`${this.baseUrl}/models`, {
        method: "GET",
        headers: this.headers(),
      });
      const json = (await resp.json()) as { data?: { id: string }[] };
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

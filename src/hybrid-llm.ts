/**
 * hybrid-llm.ts - Combines local LlamaCpp (generate/expandQuery) with remote LLM (embed/rerank).
 *
 * When QMD_REMOTE_URL is set, this class routes:
 * - embed/embedBatch/rerank → remote server (e.g. omlx with bge-m3, bge-reranker)
 * - generate/expandQuery → local LlamaCpp (QMD fine-tuned model)
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

export class HybridLLM implements LLM {
  readonly isRemote = true;

  constructor(
    private local: LLM,
    private remote: LLM,
  ) {}

  getLocal(): LLM {
    return this.local;
  }

  private getTokenizerCapableLocal(): {
    tokenize(text: string): Promise<readonly unknown[]>;
    detokenize?(tokens: readonly unknown[]): Promise<string>;
    countTokens?(text: string): Promise<number>;
  } {
    const local = this.local as LLM & {
      tokenize?: (text: string) => Promise<readonly unknown[]>;
      detokenize?: (tokens: readonly unknown[]) => Promise<string>;
      countTokens?: (text: string) => Promise<number>;
    };

    if (typeof local.tokenize !== "function") {
      throw new Error("HybridLLM local provider does not support tokenization");
    }

    return local as {
      tokenize(text: string): Promise<readonly unknown[]>;
      detokenize?(tokens: readonly unknown[]): Promise<string>;
      countTokens?(text: string): Promise<number>;
    };
  }

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    return this.remote.embed(text, options);
  }

  async embedBatch(texts: string[], options?: EmbedOptions): Promise<(EmbeddingResult | null)[]> {
    return this.remote.embedBatch(texts, options);
  }

  async rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult> {
    return this.remote.rerank(query, documents, options);
  }

  async generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    return this.local.generate(prompt, options);
  }

  async expandQuery(query: string, options?: { context?: string; includeLexical?: boolean; intent?: string }): Promise<Queryable[]> {
    return this.local.expandQuery(query, options);
  }

  async tokenize(text: string): Promise<readonly unknown[]> {
    return this.getTokenizerCapableLocal().tokenize(text);
  }

  async countTokens(text: string): Promise<number> {
    const local = this.getTokenizerCapableLocal();
    if (typeof local.countTokens === "function") {
      return local.countTokens(text);
    }
    const tokens = await local.tokenize(text);
    return tokens.length;
  }

  async detokenize(tokens: readonly unknown[]): Promise<string> {
    const local = this.getTokenizerCapableLocal();
    if (typeof local.detokenize !== "function") {
      throw new Error("HybridLLM local provider does not support detokenization");
    }
    return local.detokenize(tokens);
  }

  async modelExists(model: string): Promise<ModelInfo> {
    // Try remote first, fallback to local
    const remoteResult = await this.remote.modelExists(model);
    if (remoteResult.exists) return remoteResult;
    return this.local.modelExists(model);
  }

  async dispose(): Promise<void> {
    await Promise.all([this.local.dispose(), this.remote.dispose()]);
  }
}

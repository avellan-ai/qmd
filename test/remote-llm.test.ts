import { describe, test, expect, beforeAll, afterAll, beforeEach, vi } from "vitest";
import { RemoteLLM, type RemoteLLMConfig } from "../src/remote-llm.js";
import { HybridLLM } from "../src/hybrid-llm.js";
import { FullRemoteLLM, type FullRemoteLLMConfig } from "../src/full-remote-llm.js";
import { LlamaCpp, getDefaultLlamaCpp, setDefaultLLM } from "../src/llm.js";
import http from "http";

// =============================================================================
// Mock HTTP Server
// =============================================================================

let server: http.Server;
let baseUrl: string;

// Track last requests for assertions
let lastRequest: { path: string; body: any } | null = null;

function listen(server: http.Server): Promise<string> {
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address();
      if (typeof addr === "object" && addr) {
        resolve(`http://127.0.0.1:${addr.port}/v1`);
      }
    });
  });
}

function closeServer(server: http.Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.close((error) => {
      if (error) reject(error);
      else resolve();
    });
  });
}

function createMockServer(): Promise<{ server: http.Server; baseUrl: string }> {
  return new Promise((resolve) => {
    const srv = http.createServer((req, res) => {
      let body = "";
      req.on("data", (chunk) => (body += chunk));
      req.on("end", () => {
        const parsed = body ? JSON.parse(body) : {};
        lastRequest = { path: req.url || "", body: parsed };

        res.setHeader("Content-Type", "application/json");

        if (req.url === "/v1/embeddings") {
          const input = parsed.input as string[];
          const data = input.map((text: string, index: number) => ({
            embedding: Array.from({ length: 768 }, (_, i) => Math.sin(i + text.length)),
            index,
          }));
          res.end(JSON.stringify({ data, model: parsed.model }));
        } else if (req.url === "/v1/chat/completions") {
          const messages = parsed.messages as Array<{ role: string; content: string }>;
          const system = messages.find((m) => m.role === "system")?.content ?? "";
          const user = messages.filter((m) => m.role === "user").at(-1)?.content ?? "";
          const content = system.includes("search query expansion assistant")
            ? [
                `lex: ${user} keywords`,
                `lex: "${user} phrase"`,
                `vec: semantic ${user} explanation`,
                `vec: related ${user} concept`,
                `hyde: Information about ${user} in an ideal document`,
              ].join("\n")
            : `Generated response for ${user}`;
          res.end(JSON.stringify({
            model: parsed.model,
            choices: [
              {
                message: {
                  content,
                },
              },
            ],
          }));
        } else if (req.url === "/v1/rerank") {
          const docs = parsed.documents as string[];
          const results = docs.map((_: string, index: number) => ({
            index,
            relevance_score: 1 - index * 0.1,
          }));
          res.end(JSON.stringify({ results }));
        } else if (req.url === "/v1/models") {
          res.end(JSON.stringify({
            data: [
              { id: "bge-m3" },
              { id: "bge-reranker-v2-m3" },
              { id: "accounts/fireworks/models/qwen3-8b" },
              { id: "chat-model" },
            ],
          }));
        } else {
          res.statusCode = 404;
          res.end(JSON.stringify({ error: "not found" }));
        }
      });
    });

    srv.listen(0, "127.0.0.1", () => {
      const addr = srv.address();
      if (typeof addr === "object" && addr) {
        resolve({ server: srv, baseUrl: `http://127.0.0.1:${addr.port}/v1` });
      }
    });
  });
}

// =============================================================================
// Tests
// =============================================================================

describe("RemoteLLM", () => {
  beforeAll(async () => {
    const mock = await createMockServer();
    server = mock.server;
    baseUrl = mock.baseUrl;
  });

  afterAll(() => {
    server?.close();
  });

  beforeEach(() => {
    lastRequest = null;
  });

  function createRemote(overrides?: Partial<RemoteLLMConfig>): RemoteLLM {
    return new RemoteLLM({
      baseUrl,
      embedModel: "bge-m3",
      rerankModel: "bge-reranker-v2-m3",
      ...overrides,
    });
  }

  test("isRemote is true", () => {
    const remote = createRemote();
    expect(remote.isRemote).toBe(true);
  });

  // ── Embedding ──────────────────────────────────────────────────────────

  test("embed() returns embedding for single text", async () => {
    const remote = createRemote();
    const result = await remote.embed("hello world");
    expect(result).not.toBeNull();
    expect(result!.embedding).toHaveLength(768);
    expect(result!.model).toBe("bge-m3");
  });

  test("embed() uses caller-provided model override", async () => {
    const remote = createRemote();
    const result = await remote.embed("hello world", { model: "custom-embed-model" });

    expect(result).not.toBeNull();
    expect(result!.model).toBe("custom-embed-model");
    expect(lastRequest?.body.model).toBe("custom-embed-model");
  });

  test("embedBatch() sends correct request format", async () => {
    const remote = createRemote();
    const texts = ["hello", "world", "test"];
    const results = await remote.embedBatch(texts);

    expect(results).toHaveLength(3);
    expect(results.every((r) => r !== null)).toBe(true);
    expect(lastRequest?.path).toBe("/v1/embeddings");
    expect(lastRequest?.body.model).toBe("bge-m3");
    expect(lastRequest?.body.input).toEqual(texts);
  });

  test("embedBatch() uses caller-provided model override", async () => {
    const remote = createRemote();
    const results = await remote.embedBatch(["hello", "world"], { model: "batch-override-model" });

    expect(results).toHaveLength(2);
    expect(results.every((r) => r?.model === "batch-override-model")).toBe(true);
    expect(lastRequest?.body.model).toBe("batch-override-model");
  });

  test("embedBatch() returns empty array for empty input", async () => {
    const remote = createRemote();
    const results = await remote.embedBatch([]);
    expect(results).toEqual([]);
  });

  test("embedBatch() preserves order via index field", async () => {
    const remote = createRemote();
    const results = await remote.embedBatch(["short", "a longer piece of text"]);
    expect(results).toHaveLength(2);
    // Different input lengths produce different embeddings
    expect(results[0]!.embedding).not.toEqual(results[1]!.embedding);
  });

  test("embedBatch() sanitizes empty strings and unpaired surrogates before sending", async () => {
    let capturedInput: string[] = [];
    const embedServer = http.createServer((req, res) => {
      let body = "";
      req.on("data", (chunk) => (body += chunk));
      req.on("end", () => {
        const parsed = JSON.parse(body);
        capturedInput = parsed.input;
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({
          data: capturedInput.map((text: string, index: number) => ({
            embedding: [text.length, index],
            index,
          })),
          model: parsed.model,
        }));
      });
    });
    const embedBaseUrl = await listen(embedServer);

    try {
      const remote = createRemote({ baseUrl: embedBaseUrl });
      const results = await remote.embedBatch(["", "\uD800test", "ok\uDC00"]);

      expect(results).toHaveLength(3);
      expect(capturedInput).toEqual([" ", "test", "ok"]);
      expect(capturedInput.every((text) => !/[\uD800-\uDFFF]/.test(text))).toBe(true);
    } finally {
      await closeServer(embedServer);
    }
  });

  // ── Reranking ──────────────────────────────────────────────────────────

  test("rerank() sends correct request format", async () => {
    const remote = createRemote();
    const docs = [
      { file: "a.md", text: "first document" },
      { file: "b.md", text: "second document" },
    ];
    const result = await remote.rerank("test query", docs);

    expect(lastRequest?.path).toBe("/v1/rerank");
    expect(lastRequest?.body.model).toBe("bge-reranker-v2-m3");
    expect(lastRequest?.body.query).toBe("test query");
    expect(lastRequest?.body.documents).toEqual(["first document", "second document"]);
    expect(lastRequest?.body.return_documents).toBe(false);

    expect(result.results).toHaveLength(2);
    expect(result.results[0]!.file).toBe("a.md");
    expect(result.results[0]!.score).toBeGreaterThan(0);
    expect(result.model).toBe("bge-reranker-v2-m3");
  });

  test("rerank() returns empty results for empty documents", async () => {
    const remote = createRemote();
    const result = await remote.rerank("query", []);
    expect(result.results).toEqual([]);
  });

  test("rerank() supports Fireworks-style data responses", async () => {
    const fireworksServer = http.createServer((req, res) => {
      let body = "";
      req.on("data", (chunk) => (body += chunk));
      req.on("end", () => {
        res.setHeader("Content-Type", "application/json");
        if (req.url === "/v1/rerank") {
          res.end(JSON.stringify({
            data: [
              { index: 1, relevance_score: 0.97 },
              { index: 0, relevance_score: 0.42 },
            ],
          }));
          return;
        }
        res.statusCode = 404;
        res.end(JSON.stringify({ error: "not found" }));
      });
    });
    const fireworksBaseUrl = await listen(fireworksServer);

    try {
      const remote = createRemote({ baseUrl: fireworksBaseUrl });
      const docs = [
        { file: "alpha.md", text: "alpha content" },
        { file: "beta.md", text: "beta content" },
      ];

      const result = await remote.rerank("auth setup", docs);

      expect(result.results).toEqual([
        { file: "beta.md", index: 1, score: 0.97 },
        { file: "alpha.md", index: 0, score: 0.42 },
      ]);
      expect(result.model).toBe("bge-reranker-v2-m3");
    } finally {
      await closeServer(fireworksServer);
    }
  });

  test("rerank() throws when the server exceeds rerankTimeoutMs", async () => {
    const slowServer = http.createServer((req, res) => {
      if (req.url !== "/v1/rerank") {
        res.statusCode = 404;
        res.end(JSON.stringify({ error: "not found" }));
        return;
      }

      req.on("data", () => {});
      req.on("end", () => {
        setTimeout(() => {
          if (!res.writableEnded) {
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ results: [{ index: 0, relevance_score: 1 }] }));
          }
        }, 200);
      });
    });
    const slowBaseUrl = await listen(slowServer);

    try {
      const remote = createRemote({ baseUrl: slowBaseUrl, rerankTimeoutMs: 50 });

      let caught: unknown;
      try {
        await remote.rerank("query", [{ file: "a.md", text: "doc" }]);
      } catch (error) {
        caught = error;
      }

      expect(caught).toBeTruthy();
      expect(String(caught)).toMatch(/abort/i);
    } finally {
      await closeServer(slowServer);
    }
  });

  test("rerank() rethrows remote failures instead of returning zero scores", async () => {
    const failingServer = http.createServer((req, res) => {
      res.statusCode = req.url === "/v1/rerank" ? 503 : 200;
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ error: "rerank unavailable" }));
    });
    const failingBaseUrl = await listen(failingServer);

    try {
      const remote = createRemote({ baseUrl: failingBaseUrl });
      await expect(
        remote.rerank("query", [{ file: "a.md", text: "doc" }]),
      ).rejects.toThrow("Remote LLM error 503: {\"error\":\"rerank unavailable\"}");
    } finally {
      await closeServer(failingServer);
    }
  });

  test("rerank() truncates upstream error bodies in thrown errors", async () => {
    const longBody = "x".repeat(250);
    const failingServer = http.createServer((_req, res) => {
      res.statusCode = 500;
      res.setHeader("Content-Type", "text/plain");
      res.end(longBody);
    });
    const failingBaseUrl = await listen(failingServer);

    try {
      const remote = createRemote({ baseUrl: failingBaseUrl });
      await expect(
        remote.rerank("query", [{ file: "a.md", text: "doc" }]),
      ).rejects.toThrow(`Remote LLM error 500: ${"x".repeat(200)}`);
      await expect(
        remote.rerank("query", [{ file: "a.md", text: "doc" }]),
      ).rejects.not.toThrow("x".repeat(201));
    } finally {
      await closeServer(failingServer);
    }
  });

  test("embedBatch() caps individual retries and skips the remainder", async () => {
    const requestTimes: number[] = [];
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const failingServer = http.createServer((req, res) => {
      if (req.url === "/v1/embeddings") {
        requestTimes.push(Date.now());
      }
      res.statusCode = 429;
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ error: "rate limited" }));
    });
    const failingBaseUrl = await listen(failingServer);

    try {
      const remote = createRemote({ baseUrl: failingBaseUrl });
      const start = Date.now();
      const results = await remote.embedBatch(Array.from({ length: 12 }, (_, i) => `chunk-${i}`));
      const elapsed = Date.now() - start;

      expect(results).toHaveLength(12);
      expect(results.every((result) => result === null)).toBe(true);
      expect(requestTimes).toHaveLength(11);
      expect(elapsed).toBeGreaterThanOrEqual(850);
      expect(warnSpy).toHaveBeenCalledWith(
        "[remote-llm] individual embed retry cap hit at 10; skipping remaining 2 items",
      );
    } finally {
      warnSpy.mockRestore();
      await closeServer(failingServer);
    }
  });

  // ── Model Exists ───────────────────────────────────────────────────────

  test("modelExists() returns true for available model", async () => {
    const remote = createRemote();
    const result = await remote.modelExists("bge-m3");
    expect(result.exists).toBe(true);
    expect(result.name).toBe("bge-m3");
  });

  test("modelExists() returns false for unknown model", async () => {
    const remote = createRemote();
    const result = await remote.modelExists("nonexistent-model");
    expect(result.exists).toBe(false);
  });

  test("modelExists() returns false when the server returns non-200", async () => {
    const failingServer = http.createServer((_req, res) => {
      res.statusCode = 500;
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ error: "boom" }));
    });
    const failingBaseUrl = await listen(failingServer);

    try {
      const remote = createRemote({ baseUrl: failingBaseUrl });
      await expect(remote.modelExists("bge-m3")).resolves.toEqual({ name: "bge-m3", exists: false });
    } finally {
      await closeServer(failingServer);
    }
  });

  test("modelExists() returns false when the server times out", async () => {
    const slowServer = http.createServer((_req, res) => {
      setTimeout(() => {
        if (!res.writableEnded) {
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ data: [{ id: "bge-m3" }] }));
        }
      }, 200);
    });
    const slowBaseUrl = await listen(slowServer);

    try {
      const remote = createRemote({ baseUrl: slowBaseUrl, timeoutMs: 50 });
      await expect(remote.modelExists("bge-m3")).resolves.toEqual({ name: "bge-m3", exists: false });
    } finally {
      await closeServer(slowServer);
    }
  });

  test("modelExists() returns false when the server returns malformed JSON", async () => {
    const malformedServer = http.createServer((_req, res) => {
      res.setHeader("Content-Type", "application/json");
      res.end("{bad json");
    });
    const malformedBaseUrl = await listen(malformedServer);

    try {
      const remote = createRemote({ baseUrl: malformedBaseUrl });
      await expect(remote.modelExists("bge-m3")).resolves.toEqual({ name: "bge-m3", exists: false });
    } finally {
      await closeServer(malformedServer);
    }
  });

  test("constructor rejects non-http base URLs", () => {
    expect(() => createRemote({ baseUrl: "ftp://example.com/v1" })).toThrow(
      "Remote LLM baseUrl must start with http:// or https://",
    );
  });

  // ── Unsupported Operations ─────────────────────────────────────────────

  test("generate() throws not supported", async () => {
    const remote = createRemote();
    await expect(remote.generate("hello")).rejects.toThrow("not support");
  });

  test("expandQuery() throws not supported", async () => {
    const remote = createRemote();
    await expect(remote.expandQuery("test")).rejects.toThrow("not support");
  });

  // ── Auth Header ────────────────────────────────────────────────────────

  test("sends Authorization header when apiKey is set", async () => {
    // Create a server that captures headers
    let capturedAuth: string | undefined;
    const authServer = http.createServer((req, res) => {
      capturedAuth = req.headers.authorization;
      let body = "";
      req.on("data", (chunk) => (body += chunk));
      req.on("end", () => {
        res.setHeader("Content-Type", "application/json");
        const parsed = JSON.parse(body);
        const input = parsed.input as string[];
        res.end(JSON.stringify({
          data: input.map((_, i: number) => ({ embedding: [0, 1, 2], index: i })),
          model: "test",
        }));
      });
    });

    await new Promise<void>((resolve) => authServer.listen(0, "127.0.0.1", resolve));
    const addr = authServer.address();
    const port = typeof addr === "object" && addr ? addr.port : 0;

    try {
      const remote = new RemoteLLM({
        baseUrl: `http://127.0.0.1:${port}/v1`,
        apiKey: "test-key-123",
      });
      await remote.embed("test");
      expect(capturedAuth).toBe("Bearer test-key-123");
    } finally {
      authServer.close();
    }
  });

  // ── Dispose ────────────────────────────────────────────────────────────

  test("dispose() is a no-op", async () => {
    const remote = createRemote();
    await expect(remote.dispose()).resolves.toBeUndefined();
  });
});

// =============================================================================
// FullRemoteLLM Tests
// =============================================================================

describe("FullRemoteLLM", () => {
  beforeAll(async () => {
    const mock = await createMockServer();
    server = mock.server;
    baseUrl = mock.baseUrl;
  });

  afterAll(() => {
    server?.close();
  });

  beforeEach(() => {
    lastRequest = null;
  });

  function createFullRemote(overrides?: Partial<FullRemoteLLMConfig>): FullRemoteLLM {
    return new FullRemoteLLM({
      baseUrl,
      embedModel: "bge-m3",
      rerankModel: "bge-reranker-v2-m3",
      expandModel: "accounts/fireworks/models/qwen3-8b",
      generateModel: "chat-model",
      ...overrides,
    });
  }

  test("isRemote is true", () => {
    const remote = createFullRemote();
    expect(remote.isRemote).toBe(true);
  });

  test("generate() uses remote chat completions", async () => {
    const remote = createFullRemote();
    const result = await remote.generate("hello remote", { maxTokens: 50, temperature: 0.2 });

    expect(result).not.toBeNull();
    expect(result!.text).toContain("hello remote");
    expect(result!.model).toBe("chat-model");
    expect(lastRequest?.path).toBe("/v1/chat/completions");
    expect(lastRequest?.body.model).toBe("chat-model");
    expect(lastRequest?.body.max_tokens).toBe(50);
    expect(lastRequest?.body.temperature).toBe(0.2);
  });

  test("expandQuery() parses lex vec and hyde lines from remote chat", async () => {
    const remote = createFullRemote();
    const result = await remote.expandQuery("auth setup");

    expect(lastRequest?.path).toBe("/v1/chat/completions");
    expect(lastRequest?.body.model).toBe("accounts/fireworks/models/qwen3-8b");
    expect(result.some((q) => q.type === "lex")).toBe(true);
    expect(result.some((q) => q.type === "vec")).toBe(true);
    expect(result.some((q) => q.type === "hyde")).toBe(true);
    expect(result.every((q) => /auth|setup/i.test(q.text))).toBe(true);
  });

  test("expandQuery() respects includeLexical=false", async () => {
    const remote = createFullRemote();
    const result = await remote.expandQuery("auth setup", { includeLexical: false });
    expect(result.some((q) => q.type === "lex")).toBe(false);
  });

  test("modelExists() returns true for chat models", async () => {
    const remote = createFullRemote();
    const result = await remote.modelExists("chat-model");
    expect(result.exists).toBe(true);
  });

  test("constructor rejects non-http base URLs", () => {
    expect(() =>
      createFullRemote({ baseUrl: "ssh://example.com/v1" }),
    ).toThrow("Remote LLM baseUrl must start with http:// or https://");
  });

  test("dispose() is a no-op", async () => {
    const remote = createFullRemote();
    await expect(remote.dispose()).resolves.toBeUndefined();
  });
});

// =============================================================================
// HybridLLM Tests
// =============================================================================

describe("HybridLLM", () => {
  let mockServer: http.Server;
  let mockBaseUrl: string;

  beforeAll(async () => {
    const mock = await createMockServer();
    mockServer = mock.server;
    mockBaseUrl = mock.baseUrl;
  });

  afterAll(() => {
    mockServer?.close();
  });

  test("isRemote is true", () => {
    const remote = new RemoteLLM({ baseUrl: mockBaseUrl });
    const local = {
      embed: async () => null,
      embedBatch: async () => [],
      generate: async () => ({ text: "gen", model: "local", done: true }),
      expandQuery: async () => [{ type: "vec" as const, text: "expanded" }],
      rerank: async () => ({ results: [], model: "local" }),
      modelExists: async () => ({ name: "local", exists: true }),
      dispose: async () => {},
    };
    const hybrid = new HybridLLM(local, remote);
    expect(hybrid.isRemote).toBe(true);
  });

  test("routes embed to remote", async () => {
    const remote = new RemoteLLM({ baseUrl: mockBaseUrl, embedModel: "bge-m3" });
    const localCalled = { embed: false };
    const local = {
      embed: async () => { localCalled.embed = true; return null; },
      embedBatch: async () => [],
      generate: async () => null,
      expandQuery: async () => [],
      rerank: async () => ({ results: [], model: "local" }),
      modelExists: async () => ({ name: "local", exists: false }),
      dispose: async () => {},
    };
    const hybrid = new HybridLLM(local, remote);

    const result = await hybrid.embed("test");
    expect(result).not.toBeNull();
    expect(result!.model).toBe("bge-m3");
    expect(localCalled.embed).toBe(false);
  });

  test("routes embedBatch to remote", async () => {
    const remote = new RemoteLLM({ baseUrl: mockBaseUrl });
    const local = {
      embed: async () => null,
      embedBatch: async () => [],
      generate: async () => null,
      expandQuery: async () => [],
      rerank: async () => ({ results: [], model: "local" }),
      modelExists: async () => ({ name: "local", exists: false }),
      dispose: async () => {},
    };
    const hybrid = new HybridLLM(local, remote);
    const results = await hybrid.embedBatch(["a", "b"]);
    expect(results).toHaveLength(2);
  });

  test("routes rerank to remote", async () => {
    const remote = new RemoteLLM({ baseUrl: mockBaseUrl, rerankModel: "bge-reranker-v2-m3" });
    const local = {
      embed: async () => null,
      embedBatch: async () => [],
      generate: async () => null,
      expandQuery: async () => [],
      rerank: async () => ({ results: [], model: "local" }),
      modelExists: async () => ({ name: "local", exists: false }),
      dispose: async () => {},
    };
    const hybrid = new HybridLLM(local, remote);
    const result = await hybrid.rerank("query", [{ file: "a.md", text: "doc" }]);
    expect(result.model).toBe("bge-reranker-v2-m3");
  });

  test("routes generate to local", async () => {
    const remote = new RemoteLLM({ baseUrl: mockBaseUrl });
    const local = {
      embed: async () => null,
      embedBatch: async () => [],
      generate: async () => ({ text: "local-generated", model: "qwen", done: true }),
      expandQuery: async () => [],
      rerank: async () => ({ results: [], model: "local" }),
      modelExists: async () => ({ name: "local", exists: true }),
      dispose: async () => {},
    };
    const hybrid = new HybridLLM(local, remote);
    const result = await hybrid.generate("test prompt");
    expect(result).not.toBeNull();
    expect(result!.text).toBe("local-generated");
  });

  test("routes expandQuery to local", async () => {
    const remote = new RemoteLLM({ baseUrl: mockBaseUrl });
    const local = {
      embed: async () => null,
      embedBatch: async () => [],
      generate: async () => null,
      expandQuery: async () => [{ type: "vec" as const, text: "expanded-locally" }],
      rerank: async () => ({ results: [], model: "local" }),
      modelExists: async () => ({ name: "local", exists: true }),
      dispose: async () => {},
    };
    const hybrid = new HybridLLM(local, remote);
    const results = await hybrid.expandQuery("test");
    expect(results).toHaveLength(1);
    expect(results[0]!.text).toBe("expanded-locally");
  });

  test("passes intent through to the local expandQuery implementation", async () => {
    const remote = new RemoteLLM({ baseUrl: mockBaseUrl });
    let capturedOptions: { context?: string; includeLexical?: boolean; intent?: string } | undefined;
    const local = {
      embed: async () => null,
      embedBatch: async () => [],
      generate: async () => null,
      expandQuery: async (_query: string, options?: { context?: string; includeLexical?: boolean; intent?: string }) => {
        capturedOptions = options;
        return [{ type: "vec" as const, text: "expanded-locally" }];
      },
      rerank: async () => ({ results: [], model: "local" }),
      modelExists: async () => ({ name: "local", exists: true }),
      dispose: async () => {},
    };
    const hybrid = new HybridLLM(local, remote);

    await hybrid.expandQuery("auth", { intent: "find auth config" });

    expect(capturedOptions).toEqual({ intent: "find auth config" });
  });

  test("modelExists tries remote first then local", async () => {
    const remote = new RemoteLLM({ baseUrl: mockBaseUrl });
    const local = {
      embed: async () => null,
      embedBatch: async () => [],
      generate: async () => null,
      expandQuery: async () => [],
      rerank: async () => ({ results: [], model: "local" }),
      modelExists: async (model: string) => ({ name: model, exists: model === "local-only-model" }),
      dispose: async () => {},
    };
    const hybrid = new HybridLLM(local, remote);

    // Remote has bge-m3
    const remoteModel = await hybrid.modelExists("bge-m3");
    expect(remoteModel.exists).toBe(true);

    // Falls back to local
    const localModel = await hybrid.modelExists("local-only-model");
    expect(localModel.exists).toBe(true);
  });

  test("dispose() calls both", async () => {
    let localDisposed = false;
    let remoteDisposed = false;
    const remote = {
      embed: async () => null,
      embedBatch: async () => [],
      generate: async () => null,
      expandQuery: async () => [],
      rerank: async () => ({ results: [] as any[], model: "remote" }),
      modelExists: async () => ({ name: "remote", exists: false }),
      dispose: async () => { remoteDisposed = true; },
      isRemote: true as const,
    };
    const local = {
      embed: async () => null,
      embedBatch: async () => [],
      generate: async () => null,
      expandQuery: async () => [],
      rerank: async () => ({ results: [] as any[], model: "local" }),
      modelExists: async () => ({ name: "local", exists: false }),
      dispose: async () => { localDisposed = true; },
    };
    const hybrid = new HybridLLM(local, remote);
    await hybrid.dispose();
    expect(localDisposed).toBe(true);
    expect(remoteDisposed).toBe(true);
  });

  test("exposes the underlying local LLM", () => {
    const remote = new RemoteLLM({ baseUrl: mockBaseUrl });
    const local = new LlamaCpp({});
    const hybrid = new HybridLLM(local, remote);

    expect(hybrid.getLocal()).toBe(local);
  });

  test("delegates tokenization helpers to the local LLM when available", async () => {
    const remote = new RemoteLLM({ baseUrl: mockBaseUrl });
    const local = {
      embed: async () => null,
      embedBatch: async () => [],
      generate: async () => null,
      expandQuery: async () => [],
      rerank: async () => ({ results: [], model: "local" }),
      modelExists: async () => ({ name: "local", exists: true }),
      dispose: async () => {},
      tokenize: async (text: string) => text.split("").map((_, index) => index),
      countTokens: async (text: string) => text.length,
      detokenize: async (tokens: readonly unknown[]) => `count:${tokens.length}`,
    };
    const hybrid = new HybridLLM(local, remote);

    await expect(hybrid.tokenize("abc")).resolves.toEqual([0, 1, 2]);
    await expect(hybrid.countTokens("abcd")).resolves.toBe(4);
    await expect(hybrid.detokenize([1, 2, 3])).resolves.toBe("count:3");
  });
});

describe("getDefaultLlamaCpp", () => {
  test("LlamaCpp instances are marked as local", () => {
    expect(new LlamaCpp({}).isRemote).toBe(false);
  });

  test("unwraps the local LlamaCpp from HybridLLM", () => {
    const local = new LlamaCpp({});
    const remote = {
      embed: async () => null,
      embedBatch: async () => [],
      generate: async () => null,
      expandQuery: async () => [],
      rerank: async () => ({ results: [] as any[], model: "remote" }),
      modelExists: async () => ({ name: "remote", exists: false }),
      dispose: async () => {},
      isRemote: true as const,
    };

    setDefaultLLM(new HybridLLM(local, remote));
    try {
      expect(getDefaultLlamaCpp()).toBe(local);
    } finally {
      setDefaultLLM(null);
    }
  });

  test("throws a descriptive error for fully remote defaults", () => {
    setDefaultLLM({
      embed: async () => null,
      embedBatch: async () => [],
      generate: async () => null,
      expandQuery: async () => [],
      rerank: async () => ({ results: [] as any[], model: "remote" }),
      modelExists: async () => ({ name: "remote", exists: true }),
      dispose: async () => {},
      isRemote: true,
    });

    try {
      expect(() => getDefaultLlamaCpp()).toThrow(/does not expose a local LlamaCpp instance/i);
    } finally {
      setDefaultLLM(null);
    }
  });
});

/**
 * StreamingLLM -- Frontend SSE client for consuming streamed LLM responses.
 */
export interface StreamOptions {
  url: string; model: string; messages: Array<{ role: string; content: string }>;
  provider?: "ollama" | "openai";
  onToken: (token: string) => void;
  onDone?: (fullText: string) => void;
  onError?: (err: Error) => void;
  signal?: AbortSignal;
}

export async function streamChat(opts: StreamOptions): Promise<string> {
  let fullText = "";
  const resp = await fetch(opts.url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: opts.model, messages: opts.messages, provider: opts.provider || "ollama" }),
    signal: opts.signal
  });
  if (!resp.ok) { const err = new Error(`Stream failed: ${resp.status}`); opts.onError?.(err); throw err; }
  const reader = resp.body?.getReader();
  if (!reader) throw new Error("No readable stream");
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("

");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const data = JSON.parse(line.slice(6));
        if (data.done) { opts.onDone?.(fullText); return fullText; }
        if (data.token) { fullText += data.token; opts.onToken(data.token); }
      }
    }
  } catch (err) {
    if ((err as Error).name !== "AbortError") opts.onError?.(err as Error);
  }
  opts.onDone?.(fullText);
  return fullText;
}

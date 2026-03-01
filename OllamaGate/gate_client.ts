/**
 * OllamaGate -- TypeScript client for VS Code extension integration.
 * Sends approval requests to AlchemyOS backend, receives accept/deny/other.
 * Executes the decision in the Claude Code terminal.
 */

export interface GateRequest {
  tool_name: string;
  args: Record<string, any>;
  project?: string;
  git_branch?: string;
}

export interface GateResponse {
  action: "accept" | "deny" | "other";
  reason: string;
  tier: "always_accept" | "always_deny" | "ask_ollama";
  latency_ms: number;
  model?: string;
}

export interface GateClientOptions {
  backendUrl: string;
  timeout?: number;
  onDecision?: (req: GateRequest, res: GateResponse) => void;
  onError?: (err: Error) => void;
}

export class OllamaGateClient {
  private opts: Required<GateClientOptions>;
  private stats = { accepted: 0, denied: 0, other: 0, errors: 0 };

  constructor(opts: GateClientOptions) {
    this.opts = {
      timeout: 5000,
      onDecision: () => {},
      onError: () => {},
      ...opts,
    };
  }

  async review(request: GateRequest): Promise<GateResponse> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.opts.timeout);
    try {
      const resp = await fetch(`${this.opts.backendUrl}/gate/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
        signal: controller.signal,
      });
      if (!resp.ok) throw new Error(`Gate review failed: ${resp.status}`);
      const result: GateResponse = await resp.json();
      if (result.action === "accept") this.stats.accepted++;
      else if (result.action === "deny") this.stats.denied++;
      else this.stats.other++;
      this.opts.onDecision(request, result);
      return result;
    } catch (err) {
      this.stats.errors++;
      this.opts.onError(err as Error);
      return {
        action: "accept",
        reason: "gate unreachable, defaulting accept",
        tier: "always_accept",
        latency_ms: 0,
      };
    } finally {
      clearTimeout(timer);
    }
  }

  executeInTerminal(
    terminal: { sendText: (text: string, addNewline?: boolean) => void },
    response: GateResponse
  ): void {
    switch (response.action) {
      case "accept": terminal.sendText("y", true); break;
      case "deny": terminal.sendText("n", true); break;
      case "other": terminal.sendText(response.reason, true); break;
    }
  }

  getStats() { return { ...this.stats }; }
}

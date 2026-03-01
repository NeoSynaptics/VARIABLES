/**
 * TerminalCapture -- Capture and filter VS Code terminal output.
 * Requires: "enabledApiProposals": ["terminalDataWriteEvent"] in package.json
 */
import * as vscode from "vscode";

export interface CapturedChunk {
  text: string;
  timestamp: number;
  terminalName: string;
}

export interface CaptureOptions {
  nameFilter?: RegExp;
  stripAnsi?: boolean;
  bufferMs?: number;
  maxBufferSize?: number;
  onOutput: (chunk: CapturedChunk) => void;
  onClose?: (name: string) => void;
}

const ANSI_RE = /\[[0-9;]*[a-zA-Z]|\].*?|[^[].|/g;

export function stripAnsi(text: string): string {
  return text.replace(ANSI_RE, "");
}

export function createCapture(opts: CaptureOptions): vscode.Disposable {
  const disposables: vscode.Disposable[] = [];
  let buffer = "";
  let flushTimer: ReturnType<typeof setTimeout> | null = null;

  function flush(name: string) {
    if (buffer.length > 0) {
      opts.onOutput({ text: buffer, timestamp: Date.now(), terminalName: name });
      buffer = "";
    }
    if (flushTimer) { clearTimeout(flushTimer); flushTimer = null; }
  }

  const dataListener = (vscode.window as any).onDidWriteTerminalData?.(
    (e: { terminal: vscode.Terminal; data: string }) => {
      const name = e.terminal.name;
      if (opts.nameFilter && !opts.nameFilter.test(name)) return;
      let text = e.data;
      if (opts.stripAnsi) text = stripAnsi(text);
      if (!text.trim()) return;
      if (opts.bufferMs && opts.bufferMs > 0) {
        buffer += text;
        if (opts.maxBufferSize && buffer.length >= opts.maxBufferSize) flush(name);
        else if (!flushTimer) flushTimer = setTimeout(() => flush(name), opts.bufferMs);
      } else {
        opts.onOutput({ text, timestamp: Date.now(), terminalName: name });
      }
    }
  );
  if (dataListener) disposables.push(dataListener);
  if (opts.onClose) {
    disposables.push(vscode.window.onDidCloseTerminal((t: vscode.Terminal) => {
      if (!opts.nameFilter || opts.nameFilter.test(t.name)) {
        flush(t.name);
        opts.onClose!(t.name);
      }
    }));
  }
  return vscode.Disposable.from(...disposables);
}

export function sendToTerminal(terminal: vscode.Terminal, text: string, addNewline = true): void {
  terminal.sendText(text, addNewline);
}

export function findTerminal(pattern: RegExp): vscode.Terminal | undefined {
  return vscode.window.terminals.find((t: vscode.Terminal) => pattern.test(t.name));
}

import {getLlama, LlamaChatSession} from "node-llama-cpp";
import {fileURLToPath} from "url";
import path from "path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export class LlamaPrompter {
    constructor(modelRelativePath = "../models/hf_Qwen_Qwen3-1.7B.Q8_0.gguf") {
        this.modelPath = path.join(__dirname, modelRelativePath);
        this.llama = null;
        this.model = null;
        this.context = null;
        this.session = null;
    }

    async init() {
        if (this.session) return; // already initialized
        this.llama = await getLlama();
        this.model = await this.llama.loadModel({modelPath: this.modelPath});
        this.context = await this.model.createContext();
        this.session = new LlamaChatSession({
            contextSequence: this.context.getSequence(),
        });
    }

    async prompt(message) {
        if (!this.session) await this.init();
        const response = await this.session.prompt(message);
        return response.trim();
    }

    async dispose() {
        this.session?.dispose();
        this.context?.dispose();
        this.model?.dispose();
        this.llama?.dispose();
        this.session = this.context = this.model = this.llama = null;
    }
}

// OpenAIClient.js
import OpenAI from "openai";
import "dotenv/config";

export class OpenAIClient {
    constructor(apiKey = process.env.OPENAI_API_KEY, model = "gpt-4o") {
        if (!apiKey) {
            throw new Error("Missing OPENAI_API_KEY environment variable.");
        }

        this.client = new OpenAI({apiKey});
        this.model = model;
    }

    async send(input) {
        try {
            const response = await this.client.responses.create({
                model: this.model,
                input,
            });

            return response.output_text;
        } catch (error) {
            console.error("OpenAI API error:", error.message);
            throw error;
        }
    }
}

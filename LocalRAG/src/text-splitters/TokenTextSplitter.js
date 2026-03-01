import {BaseTextSplitter} from "./BaseTextSplitter.js";
import {RecursiveCharacterTextSplitter} from "./RecursiveCharacterTextSplitter.js";

export class TokenTextSplitter extends BaseTextSplitter {
    constructor({
                    encodingName = 'cl100k_base', // GPT-4 encoding
                    chunkSize = 1000,
                    chunkOverlap = 200
                } = {}) {
        const lengthFunction = text => Math.ceil(text.length / 4);
        super({chunkSize, chunkOverlap, lengthFunction});
        this.encodingName = encodingName;
    }

    splitText(text) {
        const splitter = new RecursiveCharacterTextSplitter({
            separators: ['\n\n', '\n', '. ', ' ', ''],
            chunkSize: this.chunkSize,
            chunkOverlap: this.chunkOverlap,
            lengthFunction: this.lengthFunction
        });
        return splitter.splitText(text);
    }
}
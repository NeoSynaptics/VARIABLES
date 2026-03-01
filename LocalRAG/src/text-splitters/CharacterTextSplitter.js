import {BaseTextSplitter} from "./BaseTextSplitter.js";

export class CharacterTextSplitter extends BaseTextSplitter {
    constructor({
                    separator = '\n\n',
                    chunkSize = 1000,
                    chunkOverlap = 200,
                    lengthFunction,
                    keepSeparator = false
                } = {}) {
        super({chunkSize, chunkOverlap, lengthFunction, keepSeparator});
        this.separator = separator;
    }

    splitText(text) {
        const splits = text.split(this.separator).filter(s => s.trim().length > 0);
        return this.mergeSplits(splits, this.separator);
    }
}
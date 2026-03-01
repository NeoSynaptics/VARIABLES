import {BaseTextSplitter} from "./BaseTextSplitter.js";

export class RecursiveCharacterTextSplitter extends BaseTextSplitter {
    constructor({
                    separators = ['\n\n', '\n', '. ', ' ', ''],
                    chunkSize = 1000,
                    chunkOverlap = 200,
                    lengthFunction,
                    keepSeparator = false
                } = {}) {
        super({chunkSize, chunkOverlap, lengthFunction, keepSeparator});
        this.separators = separators;
    }

    splitText(text) {
        const finalChunks = [];
        let separator = this.separators.at(-1);
        let nextSeparators = [];

        // choose appropriate separator
        for (let i = 0; i < this.separators.length; i++) {
            const sep = this.separators[i];
            if (text.includes(sep)) {
                separator = sep;
                nextSeparators = this.separators.slice(i + 1);
                break;
            }
        }

        const splits = text.split(separator).filter(Boolean);
        let temp = [];

        for (const s of splits) {
            if (this.lengthFunction(s) <= this.chunkSize) {
                temp.push(s);
            } else {
                if (temp.length) {
                    finalChunks.push(...this.mergeSplits(temp, separator));
                    temp = [];
                }

                if (nextSeparators.length === 0) {
                    finalChunks.push(s);
                } else {
                    const recursiveSplitter = new RecursiveCharacterTextSplitter({
                        ...this,
                        separators: nextSeparators
                    });
                    finalChunks.push(...recursiveSplitter.splitText(s));
                }
            }
        }

        if (temp.length) {
            finalChunks.push(...this.mergeSplits(temp, separator));
        }

        return finalChunks;
    }
}
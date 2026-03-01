import {Document, PDFLoader} from '../../src/index.js';
import {OutputHelper} from "../../helpers/output-helper.js";
import chalk from "chalk";

/**
 * Base class for text splitting logic.
 * Inspired by LangChain.js, but simplified and optimized for readability.
 */
class TextSplitter {
    constructor({
                    chunkSize = 1000,
                    chunkOverlap = 200,
                    lengthFunction = t => t.length,
                    keepSeparator = false
                } = {}) {
        if (chunkOverlap >= chunkSize) {
            throw new Error('chunkOverlap must be less than chunkSize');
        }

        Object.assign(this, {chunkSize, chunkOverlap, lengthFunction, keepSeparator});
    }

    /**
     * Splits a single text string — must be implemented by subclasses.
     */
    splitText() {
        throw new Error('splitText() must be implemented by subclass');
    }

    /**
     * Splits a list of Document objects into chunked Documents.
     */
    async splitDocuments(documents) {
        const chunks = [];
        for (const doc of documents) {
            chunks.push(...await this.createDocuments([doc.pageContent], [doc.metadata]));
        }
        return chunks;
    }

    /**
     * Converts raw text segments into Document objects with metadata.
     */
    async createDocuments(texts, metadatas = []) {
        const documents = [];
        for (let i = 0; i < texts.length; i++) {
            const text = texts[i];
            const metadata = metadatas[i] || {};
            const chunks = this.splitText(text);

            for (let j = 0; j < chunks.length; j++) {
                documents.push(
                    new Document(chunks[j], {
                        ...metadata,
                        chunk: j,
                        totalChunks: chunks.length
                    })
                );
            }
        }
        return documents;
    }

    /**
     * Merges text splits into chunks with overlap.
     */
    mergeSplits(splits, separator) {
        const chunks = [];
        let current = [];
        let length = 0;

        for (const split of splits) {
            const splitLength = this.lengthFunction(split);
            const extraLength = current.length ? separator.length : 0;

            // finalize current chunk if it exceeds size
            if (length + splitLength + extraLength > this.chunkSize) {
                if (current.length) {
                    chunks.push(this.joinSplits(current, separator));
                }

                // maintain overlap
                while (length > this.chunkOverlap && current.length) {
                    length -= this.lengthFunction(current.shift()) + separator.length;
                }
            }

            current.push(split);
            length += splitLength + (current.length > 1 ? separator.length : 0);
        }

        if (current.length) {
            chunks.push(this.joinSplits(current, separator));
        }

        return chunks.filter(Boolean);
    }

    /**
     * Joins text splits with a separator and trims whitespace.
     */
    joinSplits(splits, separator) {
        const text = splits.join(separator).trim();
        return text || null;
    }
}

/**
 * Character-based text splitter.
 * Splits text by a given separator (e.g., paragraphs or newlines).
 */
class CharacterTextSplitter extends TextSplitter {
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

/**
 * Recursive character-based text splitter.
 * Tries larger separators first, then falls back to smaller ones (paragraph → sentence → word).
 */
class RecursiveCharacterTextSplitter extends TextSplitter {
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

/**
 * Token-based text splitter.
 * Approximates token count (1 token ≈ 4 chars) for use with LLMs.
 * Simplified version
 */
class TokenTextSplitter extends TextSplitter {
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


// EXAMPLES that show the usage of the different splitters
async function example1() {
    const loader = new PDFLoader('https://arxiv.org/pdf/2402.19473', {splitPages: true});
    const documents = await OutputHelper.withSpinner('Loading PDF...', () => loader.load());

    OutputHelper.log.info('Using CharacterTextSplitter (500/50)');
    const splitter = new CharacterTextSplitter({chunkSize: 500, chunkOverlap: 50});

    const chunks = await OutputHelper.withSpinner('Splitting text...', () => splitter.splitDocuments(documents));

    OutputHelper.formatStats({
        Pages: documents.length,
        Chunks: chunks.length,
        AvgPerPage: (chunks.length / documents.length).toFixed(1),
        Splitter: 'CharacterTextSplitter'
    });
    chunks.slice(0, 3).forEach(OutputHelper.formatChunkPreview);
}

async function example2() {
    const loader = new PDFLoader('https://arxiv.org/pdf/2402.19473', {splitPages: true});
    const docs = await OutputHelper.withSpinner('Loading PDF...', () => loader.load());

    OutputHelper.log.info('Using RecursiveCharacterTextSplitter (1000/200)');
    const splitter = new RecursiveCharacterTextSplitter({chunkSize: 1000, chunkOverlap: 200});

    const chunks = await OutputHelper.withSpinner('Splitting recursively...', () => splitter.splitDocuments(docs));
    const {avg, min, max, median} = OutputHelper.analyzeChunks(chunks);

    OutputHelper.formatStats({
        'Total Chunks': chunks.length,
        'Average Size': `${avg} chars`,
        'Min Size': `${min} chars`,
        'Max Size': `${max} chars`,
        'Median Size': `${median} chars`,
        'Chunk Size Limit': '1000',
        'Overlap': '200'
    });

    OutputHelper.formatChunkPreview(chunks[9] || chunks[0], 9);
}

async function example3() {
    const loader = new PDFLoader('https://arxiv.org/pdf/2402.19473', {splitPages: true});
    const docs = await OutputHelper.withSpinner('Loading PDF...', () => loader.load());

    OutputHelper.log.info('Using TokenTextSplitter (500 tokens)');
    const splitter = new TokenTextSplitter({chunkSize: 500, chunkOverlap: 50});

    const chunks = await OutputHelper.withSpinner('Splitting by tokens...', () => splitter.splitDocuments(docs));

    const tokens = chunks.map(c => splitter.lengthFunction(c.pageContent));
    const avg = Math.round(tokens.reduce((a, b) => a + b, 0) / tokens.length);
    const min = Math.min(...tokens);
    const max = Math.max(...tokens);

    OutputHelper.formatStats({
        'Total Chunks': chunks.length,
        'Average Tokens': `~${avg}`,
        'Min Tokens': `~${min}`,
        'Max Tokens': `~${max}`,
        'Token Limit': '500',
        'Overlap': '50',
        'Note': 'Approximate (1 token ≈ 4 chars)'
    });

    OutputHelper.formatChunkPreview(chunks[0], 0);
}

async function example4() {
    const loader = new PDFLoader('https://arxiv.org/pdf/2402.19473', {splitPages: true});
    const docs = await OutputHelper.withSpinner('Loading full PDF...', () => loader.load());

    OutputHelper.log.info('Using RecursiveCharacterTextSplitter (800/100)');
    const splitter = new RecursiveCharacterTextSplitter({chunkSize: 800, chunkOverlap: 100});

    const chunks = await OutputHelper.withSpinner('Splitting single document...', () => splitter.splitDocuments(docs));
    OutputHelper.log.info('Filtering chunks by metadata...');
    const firstTen = chunks.filter(c => c.metadata.chunk < 10);
    OutputHelper.log.success(`Found ${firstTen.length} chunks in range [0–9]`);

    const query = 'retrieval augmented generation';
    console.log(`\nSimulated Retrieval:\nQuery: ${chalk.cyan(query)}`);

    const relevant = chunks.filter(c => c.pageContent.toLowerCase().includes(query)).slice(0, 3);
    OutputHelper.log.success(`Found ${relevant.length} relevant chunks`);
    relevant.forEach(OutputHelper.formatChunkPreview);
}

async function example5() {
    const loader = new PDFLoader('https://arxiv.org/pdf/2402.19473', {splitPages: true});
    const docs = await OutputHelper.withSpinner('Loading PDF...', () => loader.load());

    console.log('\nStrategy Comparison:\n');

    const strategies = [
        {name: 'Large (1500/150)', size: 1500, overlap: 150, use: 'Better context, fewer chunks'},
        {name: 'Medium (1000/200)', size: 1000, overlap: 200, use: 'Balanced approach'},
        {name: 'Small (500/50)', size: 500, overlap: 50, use: 'Better precision, more chunks'}
    ];

    for (const s of strategies) {
        const splitter = new RecursiveCharacterTextSplitter({chunkSize: s.size, chunkOverlap: s.overlap});
        const chunks = await OutputHelper.withSpinner(`Testing ${s.name}...`, () => splitter.splitDocuments(docs));
        const avg = Math.round(chunks.reduce((sum, c) => sum + c.pageContent.length, 0) / chunks.length);
        OutputHelper.formatStrategyComparison(`Strategy: ${s.name}`, chunks.length, avg, s.use);
    }

    console.log(`\n${chalk.bold('Summary:')}
  • More chunks = Better precision
  • Fewer chunks = Better context
  • Overlap = Maintains context between chunks
  • ${chalk.green('Start with Medium (1000/200)')}\n`);
}

// ============================================================================
// MAIN RUNNER
// ============================================================================

async function runAllExamples() {
    console.clear();
    console.log('\n' + '='.repeat(80));
    console.log(chalk.bold('RAG from Scratch - Text Splitting Examples'));
    console.log('='.repeat(80) + '\n');

    await OutputHelper.runExample('Example 1: Basic Character Splitting', example1);
    await OutputHelper.runExample('Example 2: Recursive Character Splitting', example2, 'Recommended approach');
    await OutputHelper.runExample('Example 3: Token-Based Splitting', example3);
    await OutputHelper.runExample('Example 4: Custom Splitting with Metadata', example4);
    await OutputHelper.runExample('Example 5: Comparing Splitting Strategies', example5);

    console.log(chalk.bold.green('\nAll examples completed successfully!\n'));
}

runAllExamples();

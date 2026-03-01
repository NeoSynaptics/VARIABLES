/**
 * BaseLoader - Abstract base class for document loaders
 *
 * Follows LangChain.js BaseLoader pattern
 * All loaders should extend this class
 */

export class BaseLoader {
    /**
     * Load documents from source
     * Must be implemented by subclasses
     * @returns {Promise<Document[]>} Array of documents
     * @throws {Error} If not implemented
     */
    async load() {
        throw new Error('load() must be implemented by subclass');
    }

    /**
     * Load and split documents in one step
     * Convenience method that matches LangChain.js pattern
     * @param {TextSplitter} splitter - Text splitter instance
     * @returns {Promise<Document[]>} Array of split documents
     */
    async loadAndSplit(splitter) {
        const docs = await this.load();
        if (!splitter) {
            return docs;
        }
        return await splitter.splitDocuments(docs);
    }

    /**
     * Lazy load documents (generator pattern)
     * Optional method for streaming large documents
     * @returns {AsyncGenerator<Document>}
     */
    async *loadLazy() {
        const docs = await this.load();
        for (const doc of docs) {
            yield doc;
        }
    }
}
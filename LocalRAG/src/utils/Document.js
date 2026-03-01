/**
 * Document - Represents a document with content and metadata
 *
 * Matches LangChain.js Document interface
 * @see https://js.langchain.com/docs/api/document/classes/Document
 */

export class Document {
    /**
     * Create a Document
     * @param {string} pageContent - The text content of the document
     * @param {object} [metadata={}] - Metadata associated with the document
     * @param id
     */
    constructor(pageContent, metadata = {}, id) {
        this.pageContent = pageContent;
        this.metadata = metadata;
        this.id = id;
    }

    /**
     * Convert document to JSON
     * @returns {object} JSON representation
     */
    toJSON() {
        return {
            pageContent: this.pageContent,
            metadata: this.metadata
        };
    }

    /**
     * Create Document from JSON
     * @param {object} json - JSON object
     * @returns {Document}
     */
    static fromJSON(json) {
        return new Document(json.pageContent, json.metadata);
    }

    /**
     * Get document length
     * @returns {number} Character count
     */
    get length() {
        return this.pageContent.length;
    }

    /**
     * String representation
     * @returns {string}
     */
    toString() {
        return `Document(pageContent="${this.pageContent.substring(0, 50)}...", metadata=${JSON.stringify(this.metadata)})`;
    }
}
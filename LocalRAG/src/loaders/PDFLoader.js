/**
 * PDFLoader - PDF Document Loader
 *
 * Extracts text from PDF files and converts to Document objects.
 * Follows LangChain.js loader patterns.
 *
 * @example
 * ```javascript
 * import { PDFLoader } from './src/loaders/PDFLoader.js';
 *
 * // Load entire PDF as one document
 * const loader = new PDFLoader('https://example.com/doc.pdf');
 * const docs = await loader.load();
 *
 * // Load entire local PDF as one document
 * const data = await fs.readFile(pathToPdf);
 * const loader = new PDFLoader({data});
 * const docs = await loader.load();
 *
 * // Load with pages split
 * const loader = new PDFLoader('https://example.com/doc.pdf', {
 *   splitPages: true
 * });
 * const docs = await loader.load();
 * ```
 */

import {PDFParse} from 'pdf-parse';
import { Document } from '../utils/Document.js';
import { BaseLoader } from './BaseLoader.js';

/**
 * Clean raw PDF text
 * @param {string} text - Raw text from PDF
 * @returns {string} Cleaned text
 */
function cleanText(text) {
    return text
        .replace(/\s+/g, ' ')           // Normalize whitespace
        .replace(/\n{3,}/g, '\n\n')     // Max 2 consecutive newlines
        .trim();
}

/**
 * Construct a Document object from PDF data
 * @param {string} text - Page content
 * @param {string} source - PDF source URL/path
 * @param {object} info - PDF info object
 * @param {object} metadata - PDF metadata
 * @param {number} totalPages - Total pages in PDF
 * @param {number} [pageNumber] - Current page number (if split)
 * @returns {Document}
 */
function constructDocument(text, source, info, metadata, totalPages, pageNumber = null) {
    const docMetadata = {
        source,
        pdf: {
            version: info.version,
            info: info.info,
            metadata: metadata,
            totalPages: totalPages
        }
    };

    if (pageNumber !== null) {
        docMetadata.page = pageNumber;
        docMetadata.loc = {
            pageNumber: pageNumber
        };
    }

    return new Document(text, docMetadata);
}

/**
 * PDFLoader class
 * Loads and extracts text from PDF files
 */
export class PDFLoader extends BaseLoader {
    /**
     * Create a PDFLoader
     * @param {string} source - URL or file path to PDF
     * @param {object} options - Loading options
     * @param {boolean} options.splitPages - Split into one document per page
     * @param {number} options.pdfjs - Use PDF.js version (not implemented)
     */
    constructor(source, options = {}) {
        super();
        this.source = source;
        this.splitPages = options.splitPages ?? false;
    }

    /**
     * Load documents from PDF
     * @returns {Promise<Document[]>} Array of documents
     */
    async load() {
        const parser = new PDFParse({ url: this.source });
        const info = await parser.getInfo();
        const pages = info.total;
        const documents = [];

        // console.log(`Extracted ${pages} document(s). First page snippet:\n`);
        const firstPageText = (await parser.getText({ partial: [1] })).text;
        // console.log(firstPageText.substring(0, 500).trim() + '...');

        if (this.splitPages) {
            for (let i = 0; i < pages; i++) {
                const rawText = (await parser.getText({ partial: [i + 1] })).text;
                const cleanedText = cleanText(rawText);

                documents.push(constructDocument(
                    cleanedText,
                    this.source,
                    info,
                    info.metadata,
                    pages,
                    i + 1
                ));
            }
        } else {
            const rawText = (await parser.getText()).text;
            const cleanedText = cleanText(rawText);

            documents.push(constructDocument(
                cleanedText,
                this.source,
                info,
                info.metadata,
                pages
            ));
        }

        return documents;
    }

    /**
     * Load and split documents in one step
     * Convenience method that matches LangChain.js pattern
     * @param {TextSplitter} splitter - Text splitter instance
     * @returns {Promise<Document[]>} Array of split documents
     */
    async loadAndSplit(splitter) {
        const docs = await this.load();
        return await splitter.splitDocuments(docs);
    }
}

// Standalone function for backward compatibility
/**
 * Extract text from PDF (legacy function)
 * @deprecated Use PDFLoader class instead
 * @param {string} url - PDF URL or path
 * @param {object} options - Options
 * @param {boolean} options.splitPages - Split by pages
 * @returns {Promise<Document[]>}
 */
export const extractTextFromPDF = async (url, { splitPages } = { splitPages: false }) => {
    const loader = new PDFLoader(url, { splitPages });
    return await loader.load();
};
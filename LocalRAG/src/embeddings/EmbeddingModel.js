/**
 * EmbeddingModel - Text-to-Vector Conversion
 *
 * Wrapper around node-llama-cpp for generating embeddings.
 * Follows LangChain.js Embeddings interface pattern.
 *
 * @example
 * ```javascript
 * import { EmbeddingModel } from './src/embeddings/EmbeddingModel.js';
 *
 * const embeddings = new EmbeddingModel({
 *   modelPath: './models/bge-small-en-v1.5.Q8_0.gguf'
 * });
 *
 * await embeddings.initialize();
 *
 * // Single text
 * const vector = await embeddings.embedQuery('Hello world');
 *
 * // Multiple documents
 * const vectors = await embeddings.embedDocuments(['doc1', 'doc2']);
 * ```
 */

import { getLlama } from 'node-llama-cpp';
import { EmbeddingCache } from './EmbeddingCache.js';

export class EmbeddingModel {
    /**
     * Create an EmbeddingModel
     * @param {object} config - Configuration options
     * @param {string} config.modelPath - Path to GGUF model file
     * @param {number} [config.dimensions=384] - Expected embedding dimensions
     * @param {string} [config.logLevel='error'] - Log level for llama.cpp
     * @param {boolean} [config.cache=true] - Enable embedding cache
     * @param {number} [config.batchSize=32] - Batch size for parallel processing
     */
    constructor(config) {
        this.modelPath = config.modelPath;
        this.dimensions = config.dimensions ?? 384;
        this.logLevel = config.logLevel ?? 'error';
        this.batchSize = config.batchSize ?? 32;

        // Initialize cache if enabled
        this.useCache = config.cache ?? true;
        this.cache = this.useCache ? new EmbeddingCache() : null;

        // State
        this.llama = null;
        this.model = null;
        this.context = null;
        this.initialized = false;
    }

    /**
     * Initialize the embedding model
     * Must be called before embedding
     * @returns {Promise<void>}
     */
    async initialize() {
        if (this.initialized) {
            return;
        }

        this.llama = await getLlama({
            logLevel: this.logLevel
        });

        this.model = await this.llama.loadModel({
            modelPath: this.modelPath
        });

        this.context = await this.model.createEmbeddingContext();
        this.initialized = true;
    }

    /**
     * Ensure model is initialized
     * @private
     * @throws {Error} If not initialized
     */
    _ensureInitialized() {
        if (!this.initialized) {
            throw new Error('EmbeddingModel not initialized. Call initialize() first.');
        }
    }

    /**
     * Embed a single query text
     * Matches LangChain.js embedQuery() method
     * @param {string} text - Text to embed
     * @returns {Promise<number[]>} Embedding vector
     */
    async embedQuery(text) {
        this._ensureInitialized();

        // Check cache first
        if (this.cache) {
            const cached = await this.cache.get(text);
            if (cached) {
                return cached;
            }
        }

        // Generate embedding
        const embedding = await this.context.getEmbeddingFor(text);
        const vector = Array.from(embedding.vector);

        // Store in cache
        if (this.cache) {
            await this.cache.set(text, vector);
        }

        return vector;
    }

    /**
     * Embed multiple documents
     * Matches LangChain.js embedDocuments() method
     * @param {string[]} texts - Array of texts to embed
     * @param {function} [onProgress] - Progress callback (current, total)
     * @returns {Promise<number[][]>} Array of embedding vectors
     */
    async embedDocuments(texts, onProgress = null) {
        this._ensureInitialized();

        const vectors = [];
        let processed = 0;

        // Process in batches
        for (let i = 0; i < texts.length; i += this.batchSize) {
            const batch = texts.slice(i, Math.min(i + this.batchSize, texts.length));

            // Process batch in parallel
            const batchVectors = await Promise.all(
                batch.map(async (text) => {
                    // Check cache first
                    if (this.cache) {
                        const cached = await this.cache.get(text);
                        if (cached) {
                            return cached;
                        }
                    }

                    // Generate embedding
                    const embedding = await this.context.getEmbeddingFor(text);
                    const vector = Array.from(embedding.vector);

                    // Store in cache
                    if (this.cache) {
                        await this.cache.set(text, vector);
                    }

                    processed++;
                    if (onProgress) {
                        onProgress(processed, texts.length);
                    }

                    return vector;
                })
            );

            vectors.push(...batchVectors);
        }

        return vectors;
    }

    /**
     * Embed a single document (alias for embedQuery)
     * @param {string} text - Text to embed
     * @returns {Promise<number[]>} Embedding vector
     */
    async embed(text) {
        return await this.embedQuery(text);
    }

    /**
     * Embed documents with metadata
     * Returns embeddings with their corresponding text and metadata
     * @param {Array<{text: string, metadata?: object}>} documents - Documents to embed
     * @param {function} [onProgress] - Progress callback
     * @returns {Promise<Array<{text: string, embedding: number[], metadata?: object}>>}
     */
    async embedDocumentsWithMetadata(documents, onProgress = null) {
        this._ensureInitialized();

        const texts = documents.map(doc => doc.text);
        const vectors = await this.embedDocuments(texts, onProgress);

        return documents.map((doc, i) => ({
            text: doc.text,
            embedding: vectors[i],
            metadata: doc.metadata
        }));
    }

    /**
     * Get embedding dimensions
     * @returns {number} Dimension count
     */
    getDimensions() {
        return this.dimensions;
    }

    /**
     * Get model info
     * @returns {object} Model information
     */
    getModelInfo() {
        return {
            modelPath: this.modelPath,
            dimensions: this.dimensions,
            initialized: this.initialized,
            cacheEnabled: this.useCache,
            cacheSize: this.cache ? this.cache.size() : 0
        };
    }

    /**
     * Clear embedding cache
     */
    clearCache() {
        if (this.cache) {
            this.cache.clear();
        }
    }

    /**
     * Get cache statistics
     * @returns {object} Cache stats
     */
    getCacheStats() {
        if (!this.cache) {
            return { enabled: false };
        }
        return this.cache.getStats();
    }

    /**
     * Save cache to disk
     * @param {string} filepath - Path to save cache
     */
    async saveCache(filepath) {
        if (this.cache) {
            await this.cache.saveToDisk(filepath);
        }
    }

    /**
     * Load cache from disk
     * @param {string} filepath - Path to load cache from
     */
    async loadCache(filepath) {
        if (this.cache) {
            await this.cache.loadFromDisk(filepath);
        }
    }

    /**
     * Calculate similarity between two texts
     * Convenience method for comparing texts
     * @param {string} text1 - First text
     * @param {string} text2 - Second text
     * @returns {Promise<number>} Cosine similarity score
     */
    async similarity(text1, text2) {
        const [vec1, vec2] = await this.embedDocuments([text1, text2]);
        return this._cosineSimilarity(vec1, vec2);
    }

    /**
     * Calculate cosine similarity between two vectors
     * @private
     * @param {number[]} vec1 - First vector
     * @param {number[]} vec2 - Second vector
     * @returns {number} Similarity score (-1 to 1)
     */
    _cosineSimilarity(vec1, vec2) {
        if (vec1.length !== vec2.length) {
            throw new Error('Vectors must have same dimensions');
        }

        let dotProduct = 0;
        let mag1 = 0;
        let mag2 = 0;

        for (let i = 0; i < vec1.length; i++) {
            dotProduct += vec1[i] * vec2[i];
            mag1 += vec1[i] * vec1[i];
            mag2 += vec2[i] * vec2[i];
        }

        mag1 = Math.sqrt(mag1);
        mag2 = Math.sqrt(mag2);

        if (mag1 === 0 || mag2 === 0) {
            return 0;
        }

        return dotProduct / (mag1 * mag2);
    }

    /**
     * Cleanup resources
     * Call when done to free memory
     */
    async cleanup() {
        if (this.context) {
            // node-llama-cpp handles cleanup automatically
            this.context = null;
        }
        if (this.model) {
            this.model = null;
        }
        if (this.llama) {
            this.llama = null;
        }
        this.initialized = false;
    }
}
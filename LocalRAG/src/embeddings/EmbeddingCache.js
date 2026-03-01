/**
 * EmbeddingCache - In-Memory and Persistent Caching for Embeddings
 *
 * Provides fast lookup for previously computed embeddings.
 * Supports both in-memory cache and disk persistence.
 *
 * @example
 * ```javascript
 * import { EmbeddingCache } from './src/embeddings/EmbeddingCache.js';
 *
 * const cache = new EmbeddingCache({ maxSize: 1000 });
 *
 * // Store embedding
 * await cache.set('Hello world', [0.1, 0.2, ...]);
 *
 * // Retrieve embedding
 * const vector = await cache.get('Hello world');
 *
 * // Save to disk
 * await cache.saveToDisk('./embeddings.cache');
 *
 * // Load from disk
 * await cache.loadFromDisk('./embeddings.cache');
 * ```
 */

import fs from 'fs/promises';
import crypto from 'crypto';

export class EmbeddingCache {
    /**
     * Create an EmbeddingCache
     * @param {object} [options={}] - Cache options
     * @param {number} [options.maxSize=10000] - Maximum cache entries
     * @param {boolean} [options.useHash=true] - Hash keys for consistent length
     * @param {string} [options.hashAlgorithm='sha256'] - Hash algorithm
     */
    constructor(options = {}) {
        this.maxSize = options.maxSize ?? 10000;
        this.useHash = options.useHash ?? true;
        this.hashAlgorithm = options.hashAlgorithm ?? 'sha256';

        // Cache storage: Map<key, {vector, timestamp, hits}>
        this.cache = new Map();

        // Statistics
        this.stats = {
            hits: 0,
            misses: 0,
            sets: 0,
            evictions: 0
        };
    }

    /**
     * Generate cache key from text
     * @private
     * @param {string} text - Input text
     * @returns {string} Cache key
     */
    _generateKey(text) {
        if (!this.useHash) {
            return text;
        }

        return crypto
            .createHash(this.hashAlgorithm)
            .update(text)
            .digest('hex');
    }

    /**
     * Get embedding from cache
     * @param {string} text - Text to lookup
     * @returns {Promise<number[]|null>} Embedding vector or null if not found
     */
    async get(text) {
        const key = this._generateKey(text);
        const entry = this.cache.get(key);

        if (entry) {
            this.stats.hits++;
            entry.hits++;
            entry.lastAccessed = Date.now();
            return entry.vector;
        }

        this.stats.misses++;
        return null;
    }

    /**
     * Store embedding in cache
     * @param {string} text - Input text
     * @param {number[]} vector - Embedding vector
     * @returns {Promise<void>}
     */
    async set(text, vector) {
        const key = this._generateKey(text);

        // Check if we need to evict
        if (this.cache.size >= this.maxSize && !this.cache.has(key)) {
            this._evictLRU();
        }

        this.cache.set(key, {
            vector: vector,
            text: this.useHash ? text : null, // Store original text if hashing
            timestamp: Date.now(),
            lastAccessed: Date.now(),
            hits: 0
        });

        this.stats.sets++;
    }

    /**
     * Evict least recently used entry
     * @private
     */
    _evictLRU() {
        let oldestKey = null;
        let oldestTime = Infinity;

        for (const [key, entry] of this.cache.entries()) {
            if (entry.lastAccessed < oldestTime) {
                oldestTime = entry.lastAccessed;
                oldestKey = key;
            }
        }

        if (oldestKey) {
            this.cache.delete(oldestKey);
            this.stats.evictions++;
        }
    }

    /**
     * Check if text exists in cache
     * @param {string} text - Text to check
     * @returns {Promise<boolean>}
     */
    async has(text) {
        const key = this._generateKey(text);
        return this.cache.has(key);
    }

    /**
     * Get cache size
     * @returns {number} Number of cached entries
     */
    size() {
        return this.cache.size;
    }

    /**
     * Clear all cache entries
     */
    clear() {
        this.cache.clear();
        this.stats = {
            hits: 0,
            misses: 0,
            sets: 0,
            evictions: 0
        };
    }

    /**
     * Get cache statistics
     * @returns {object} Cache statistics
     */
    getStats() {
        const totalRequests = this.stats.hits + this.stats.misses;
        const hitRate = totalRequests > 0
            ? (this.stats.hits / totalRequests * 100).toFixed(2)
            : 0;

        return {
            enabled: true,
            size: this.cache.size,
            maxSize: this.maxSize,
            hits: this.stats.hits,
            misses: this.stats.misses,
            sets: this.stats.sets,
            evictions: this.stats.evictions,
            hitRate: `${hitRate}%`,
            totalRequests
        };
    }

    /**
     * Get all cache keys
     * @returns {string[]} Array of cache keys
     */
    keys() {
        return Array.from(this.cache.keys());
    }

    /**
     * Get cache entries sorted by access frequency
     * @param {number} [limit=10] - Number of entries to return
     * @returns {Array<{text: string, hits: number}>} Top entries
     */
    getTopEntries(limit = 10) {
        const entries = Array.from(this.cache.entries())
            .map(([key, entry]) => ({
                key,
                text: entry.text || key.substring(0, 50),
                hits: entry.hits,
                lastAccessed: entry.lastAccessed
            }))
            .sort((a, b) => b.hits - a.hits)
            .slice(0, limit);

        return entries;
    }

    /**
     * Save cache to disk in JSON format
     * @param {string} filepath - Path to save cache
     * @returns {Promise<void>}
     */
    async saveToDisk(filepath) {
        const cacheData = {
            version: '1.0',
            timestamp: new Date().toISOString(),
            maxSize: this.maxSize,
            useHash: this.useHash,
            stats: this.stats,
            entries: Array.from(this.cache.entries()).map(([key, entry]) => ({
                key,
                text: entry.text,
                vector: entry.vector,
                timestamp: entry.timestamp,
                lastAccessed: entry.lastAccessed,
                hits: entry.hits
            }))
        };

        await fs.writeFile(
            filepath,
            JSON.stringify(cacheData, null, 2),
            'utf-8'
        );
    }

    /**
     * Load cache from disk
     * @param {string} filepath - Path to load cache from
     * @returns {Promise<void>}
     */
    async loadFromDisk(filepath) {
        const content = await fs.readFile(filepath, 'utf-8');
        const cacheData = JSON.parse(content);

        // Clear existing cache
        this.clear();

        // Restore settings
        this.maxSize = cacheData.maxSize;
        this.useHash = cacheData.useHash;
        this.stats = cacheData.stats || this.stats;

        // Restore entries
        for (const entry of cacheData.entries) {
            this.cache.set(entry.key, {
                vector: entry.vector,
                text: entry.text,
                timestamp: entry.timestamp,
                lastAccessed: entry.lastAccessed,
                hits: entry.hits
            });
        }
    }

    /**
     * Save cache to disk in compact binary format
     * More efficient than JSON for large caches
     * @param {string} filepath - Path to save cache
     * @returns {Promise<void>}
     */
    async saveToDiskBinary(filepath) {
        const entries = Array.from(this.cache.entries());

        // Calculate buffer size
        let bufferSize = 16; // Header: version(4) + maxSize(4) + entryCount(4) + dimensions(4)

        for (const [key, entry] of entries) {
            // Key length + key + vector length + vector + metadata
            bufferSize += 4 + Buffer.byteLength(key) + 4 + (entry.vector.length * 4) + 8;
        }

        const buffer = Buffer.allocUnsafe(bufferSize);
        let offset = 0;

        // Write header
        buffer.writeUInt32LE(1, offset); offset += 4; // version
        buffer.writeUInt32LE(this.maxSize, offset); offset += 4;
        buffer.writeUInt32LE(entries.length, offset); offset += 4;
        buffer.writeUInt32LE(entries[0]?.[1]?.vector.length || 384, offset); offset += 4;

        // Write entries
        for (const [key, entry] of entries) {
            // Write key
            const keyBuffer = Buffer.from(key);
            buffer.writeUInt32LE(keyBuffer.length, offset);
            offset += 4;
            keyBuffer.copy(buffer, offset);
            offset += keyBuffer.length;

            // Write vector
            buffer.writeUInt32LE(entry.vector.length, offset);
            offset += 4;
            for (const value of entry.vector) {
                buffer.writeFloatLE(value, offset);
                offset += 4;
            }

            // Write metadata (hits)
            buffer.writeUInt32LE(entry.hits, offset);
            offset += 4;
            buffer.writeUInt32LE(entry.lastAccessed, offset);
            offset += 4;
        }

        await fs.writeFile(filepath, buffer);
    }

    /**
     * Load cache from binary format
     * @param {string} filepath - Path to load cache from
     * @returns {Promise<void>}
     */
    async loadFromDiskBinary(filepath) {
        const buffer = await fs.readFile(filepath);
        let offset = 0;

        // Read header
        const version = buffer.readUInt32LE(offset); offset += 4;
        this.maxSize = buffer.readUInt32LE(offset); offset += 4;
        const entryCount = buffer.readUInt32LE(offset); offset += 4;
        const dimensions = buffer.readUInt32LE(offset); offset += 4;

        // Clear existing cache
        this.clear();

        // Read entries
        for (let i = 0; i < entryCount; i++) {
            // Read key
            const keyLength = buffer.readUInt32LE(offset);
            offset += 4;
            const key = buffer.toString('utf-8', offset, offset + keyLength);
            offset += keyLength;

            // Read vector
            const vectorLength = buffer.readUInt32LE(offset);
            offset += 4;
            const vector = [];
            for (let j = 0; j < vectorLength; j++) {
                vector.push(buffer.readFloatLE(offset));
                offset += 4;
            }

            // Read metadata
            const hits = buffer.readUInt32LE(offset);
            offset += 4;
            const lastAccessed = buffer.readUInt32LE(offset);
            offset += 4;

            this.cache.set(key, {
                vector,
                text: null,
                timestamp: Date.now(),
                lastAccessed,
                hits
            });
        }
    }

    /**
     * Get memory usage estimate
     * @returns {object} Memory usage in bytes
     */
    getMemoryUsage() {
        let totalBytes = 0;

        for (const [key, entry] of this.cache.entries()) {
            // Key size
            totalBytes += Buffer.byteLength(key);

            // Vector size (4 bytes per float)
            totalBytes += entry.vector.length * 4;

            // Metadata overhead (approximate)
            totalBytes += 32;
        }

        return {
            bytes: totalBytes,
            kilobytes: (totalBytes / 1024).toFixed(2),
            megabytes: (totalBytes / 1024 / 1024).toFixed(2)
        };
    }

    /**
     * Prune cache entries by hit count
     * Remove entries with fewer hits than threshold
     * @param {number} minHits - Minimum hit count to keep
     */
    prune(minHits = 1) {
        const keysToDelete = [];

        for (const [key, entry] of this.cache.entries()) {
            if (entry.hits < minHits) {
                keysToDelete.push(key);
            }
        }

        for (const key of keysToDelete) {
            this.cache.delete(key);
        }

        return keysToDelete.length;
    }
}
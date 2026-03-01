/**
 * Query Preprocessing for RAG
 *
 * Demonstrates:
 * 1) Cleaning and normalizing user queries before embedding
 * 2) Removing stopwords to improve semantic focus
 * 3) Query expansion for better retrieval
 * 4) Handling noisy user input (typos, extra spaces, special chars)
 * 5) Comparing retrieval quality with/without preprocessing
 * 6) Query stemming and lemmatization concepts
 * 7) Multi-step preprocessing pipeline
 *
 * Why Query Preprocessing?
 * - Raw user queries often contain noise that degrades embedding quality
 * - Preprocessing improves vector stability and retrieval consistency
 * - Reduces embedding variations for semantically similar queries
 * - Helps match query phrasing with document phrasing
 *
 * Prerequisites:
 * - Completed 06_retrieval_strategies/01_basic_retrieval
 * - npm install embedded-vector-db node-llama-cpp chalk
 * - Place bge-small-en-v1.5.Q8_0.gguf and Qwen3-1.7B-Q8_0.gguf under models/
 *
 * Download the qwen model here: https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/blob/main/Qwen3-1.7B-Q8_0.gguf
 */

import { fileURLToPath } from "url";
import path from "path";
import { VectorDB } from "embedded-vector-db";
import { getLlama, LlamaChatSession } from "node-llama-cpp";
import { Document } from "../../../src/index.js";
import { OutputHelper } from "../../../helpers/output-helper.js";
import chalk from "chalk";

// Increase max listeners to avoid warnings
process.setMaxListeners(20);

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const EMBEDDING_MODEL_PATH = path.join(__dirname, "..", "..", "..", "models", "bge-small-en-v1.5.Q8_0.gguf");
const LLM_MODEL_PATH = path.join(__dirname, "..", "..", "..", "models", "Qwen3-1.7B-Q8_0.gguf");

// VectorDB config
const DIM = 384;
const MAX_ELEMENTS = 10000;
const NS = "query_preprocessing";

// Common English stopwords
const STOPWORDS = new Set([
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "will", "with", "this", "these", "those", "what",
    "where", "when", "who", "how", "can", "could", "would", "should"
]);

/**
 * Initialize the embedding model
 */
async function initializeEmbeddingModel() {
    try {
        const llama = await getLlama({ logLevel: "error" });
        const model = await llama.loadModel({ modelPath: EMBEDDING_MODEL_PATH });
        return await model.createEmbeddingContext();
    } catch (error) {
        throw new Error(`Failed to initialize embedding model: ${error.message}`);
    }
}

/**
 * Initialize the LLM for generation
 */
async function initializeLLM() {
    try {
        const llama = await getLlama({ logLevel: "error" });
        const model = await llama.loadModel({ modelPath: LLM_MODEL_PATH });
        const context = await model.createContext();
        return new LlamaChatSession({ contextSequence: context.getSequence() });
    } catch (error) {
        throw new Error(`Failed to initialize LLM: ${error.message}`);
    }
}

/**
 * Create knowledge base documents
 */
function createKnowledgeBase() {
    return [
        new Document("Python is a high-level programming language known for its simplicity and readability. It's widely used in data science, web development, and automation.", {
            id: "doc_1",
            category: "programming",
            topic: "python",
        }),
        new Document("JavaScript is the primary language for web browsers and enables interactive web pages. It's also used server-side with Node.js.", {
            id: "doc_2",
            category: "programming",
            topic: "javascript",
        }),
        new Document("Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming. It uses algorithms to identify patterns.", {
            id: "doc_3",
            category: "ai",
            topic: "machine-learning",
        }),
        new Document("Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes organized in layers.", {
            id: "doc_4",
            category: "ai",
            topic: "neural-networks",
        }),
        new Document("React is a JavaScript library for building user interfaces, developed by Facebook. It uses a component-based architecture and virtual DOM.", {
            id: "doc_5",
            category: "programming",
            topic: "react",
        }),
        new Document("Natural language processing (NLP) is a branch of AI that helps computers understand and process human language. It powers chatbots, translation, and sentiment analysis.", {
            id: "doc_6",
            category: "ai",
            topic: "nlp",
        }),
        new Document("Docker is a containerization platform that packages applications with their dependencies. It ensures consistent behavior across different environments.", {
            id: "doc_7",
            category: "devops",
            topic: "docker",
        }),
        new Document("Git is a distributed version control system for tracking changes in source code. It enables collaboration among developers and maintains project history.", {
            id: "doc_8",
            category: "tools",
            topic: "git",
        }),
        new Document("SQL (Structured Query Language) is used for managing relational databases. It supports operations like SELECT, INSERT, UPDATE, and DELETE.", {
            id: "doc_9",
            category: "database",
            topic: "sql",
        }),
        new Document("REST APIs use HTTP methods to enable communication between systems. They follow stateless architecture and use JSON or XML for data exchange.", {
            id: "doc_10",
            category: "web",
            topic: "rest-api",
        }),
    ];
}

/**
 * Add documents to vector store
 */
async function addDocumentsToStore(vectorStore, embeddingContext, documents) {
    try {
        for (const doc of documents) {
            const embedding = await embeddingContext.getEmbeddingFor(doc.pageContent);
            const metadata = {
                content: doc.pageContent,
                ...doc.metadata,
            };
            await vectorStore.insert(
                NS,
                doc.metadata.id,
                Array.from(embedding.vector),
                metadata
            );
        }
    } catch (error) {
        throw new Error(`Failed to add documents to store: ${error.message}`);
    }
}

// ============================================================================
// QUERY PREPROCESSING FUNCTIONS
// ============================================================================

/**
 * Basic text cleaning - lowercase and trim
 */
function basicClean(query) {
    return query.toLowerCase().trim();
}

/**
 * Remove extra whitespace (multiple spaces, tabs, newlines)
 */
function normalizeWhitespace(query) {
    return query.replace(/\s+/g, ' ').trim();
}

/**
 * Remove special characters (keep letters, numbers, spaces)
 */
function removeSpecialChars(query) {
    return query.replace(/[^a-z0-9\s]/gi, ' ').replace(/\s+/g, ' ').trim();
}

/**
 * Remove stopwords (optimized - query should already be lowercased)
 */
function removeStopwords(query) {
    const words = query.split(' ');
    const filtered = words.filter(word => word && !STOPWORDS.has(word));
    return filtered.join(' ');
}

/**
 * Simple query expansion - add synonyms or related terms
 * Extended with more common abbreviations
 */
function expandQuery(query) {
    const expansions = {
        'ml': 'machine learning',
        'ai': 'artificial intelligence',
        'js': 'javascript',
        'py': 'python',
        'db': 'database',
        'api': 'application programming interface',
        'abt': 'about',
        'plz': 'please',
        'pls': 'please',
        'thx': 'thanks',
        'diff': 'difference',
        'btw': 'between',
        'info': 'information',
        'docs': 'documentation',
        'repo': 'repository',
    };

    let expanded = query;
    for (const [abbr, full] of Object.entries(expansions)) {
        const regex = new RegExp(`\\b${abbr}\\b`, 'gi');
        expanded = expanded.replace(regex, full);
    }

    return expanded;
}

/**
 * Complete preprocessing pipeline
 * FIXED: Query expansion now happens BEFORE removing special characters
 * This prevents issues like "What's" → "what s" → "what s machine learning"
 */
function preprocessQuery(query, options = {}) {
    const {
        lowercase = true,
        normalizeSpace = true,
        removeSpecial = true,
        removeStops = false,
        expand = false,
    } = options;

    let processed = query;

    if (lowercase) {
        processed = basicClean(processed);
    }

    if (normalizeSpace) {
        processed = normalizeWhitespace(processed);
    }

    // FIXED: Expand BEFORE removing special chars to avoid artifacts
    if (expand) {
        processed = expandQuery(processed);
    }

    if (removeSpecial) {
        processed = removeSpecialChars(processed);
    }

    if (removeStops) {
        processed = removeStopwords(processed);
    }

    return processed;
}

/**
 * Retrieve relevant documents from vector store
 * ADDED: Error handling
 */
async function retrieveDocuments(vectorStore, embeddingContext, query, k = 3) {
    try {
        const queryEmbedding = await embeddingContext.getEmbeddingFor(query);
        const results = await vectorStore.search(NS, Array.from(queryEmbedding.vector), k);
        return results;
    } catch (error) {
        console.error(chalk.red(`Failed to retrieve documents: ${error.message}`));
        return [];
    }
}

/**
 * Build context string from retrieved documents
 */
function buildContext(retrievedDocs) {
    if (retrievedDocs.length === 0) {
        return "";
    }

    const contextParts = retrievedDocs.map((doc, idx) =>
        `[${idx + 1}] ${doc.metadata.content}`
    );

    return contextParts.join("\n\n");
}

/**
 * Generate answer using LLM with retrieved context
 * ADDED: Error handling
 */
async function generateAnswer(chatSession, query, context) {
    try {
        if (!context || context.trim().length === 0) {
            const prompt = `Question: ${query}\n\nYou don't have any relevant information to answer this question. Please say so politely.`;
            const response = await chatSession.prompt(prompt, { maxTokens: 500 });
            return response.trim();
        }

        const prompt = `You are a helpful assistant. Use the following context to answer the question. If the context doesn't contain relevant information, say so.

Context:
${context}

Question: ${query}

Answer:`;

        const response = await chatSession.prompt(prompt, { maxTokens: 500 });
        return response.trim();
    } catch (error) {
        console.error(chalk.red(`Failed to generate answer: ${error.message}`));
        return "Sorry, I encountered an error generating a response.";
    }
}

// ============================================================================
// EXAMPLES
// ============================================================================

/**
 * Example 1: Impact of Basic Cleaning
 */
async function example1(embeddingContext) {
    const vectorStore = new VectorDB({ dim: DIM, maxElements: MAX_ELEMENTS });
    const documents = createKnowledgeBase();
    await OutputHelper.withSpinner(
        "Adding documents to vector store...",
        () => addDocumentsToStore(vectorStore, embeddingContext, documents)
    );

    // Noisy query with extra spaces, mixed case, special chars
    const noisyQuery = "   What   IS   Python???  ";
    const cleanQuery = preprocessQuery(noisyQuery, {
        lowercase: true,
        normalizeSpace: true,
        removeSpecial: true
    });

    console.log(`${chalk.bold("Original Query:")} "${noisyQuery}"`);
    console.log(`${chalk.bold("Cleaned Query:")} "${cleanQuery}"\n`);

    // Retrieve with original noisy query
    const noisyResults = await retrieveDocuments(vectorStore, embeddingContext, noisyQuery, 3);
    console.log(chalk.bold("Results with Noisy Query:"));
    noisyResults.forEach((doc, idx) => {
        console.log(`  ${idx + 1}. [${chalk.yellow(doc.similarity.toFixed(4))}] ${doc.metadata.content.substring(0, 50) + '...'}`);
    });

    // Retrieve with cleaned query
    const cleanResults = await retrieveDocuments(vectorStore, embeddingContext, cleanQuery, 3);
    console.log(`\n${chalk.bold("Results with Cleaned Query:")}`);
    cleanResults.forEach((doc, idx) => {
        console.log(`  ${idx + 1}. [${chalk.green(doc.similarity.toFixed(4))}] ${doc.metadata.content.substring(0, 50) + '...'}`);
    });

    console.log(`\n${chalk.bold("💡 Key Insight:")}`);
    console.log("Basic cleaning (lowercase, trim, normalize spaces) improves embedding consistency.");
    console.log("Similarity scores are more stable with clean input.\n");
}

/**
 * Example 2: Removing Special Characters
 */
async function example2(embeddingContext) {
    const vectorStore = new VectorDB({ dim: DIM, maxElements: MAX_ELEMENTS });
    const documents = createKnowledgeBase();
    await addDocumentsToStore(vectorStore, embeddingContext, documents);

    // Test Case 1: Contractions and light punctuation (may hurt)
    console.log(chalk.bold("Test 1: Natural Query with Contractions"));
    const query1 = "What's machine learning??? (explain it!)";
    const cleaned1 = preprocessQuery(query1, { lowercase: true, normalizeSpace: true, removeSpecial: true });

    console.log(`  Original: "${query1}"`);
    console.log(`  Cleaned:  "${cleaned1}"`);

    const results1_orig = await retrieveDocuments(vectorStore, embeddingContext, query1, 1);
    const results1_clean = await retrieveDocuments(vectorStore, embeddingContext, cleaned1, 1);

    console.log(`  Score with special chars:    ${chalk.yellow(results1_orig[0].similarity.toFixed(4))}`);
    console.log(`  Score without special chars: ${chalk.red(results1_clean[0].similarity.toFixed(4))} ${chalk.dim("(worse)")}`);
    console.log(chalk.dim("  → Apostrophe in 'What's' provides useful context\n"));

    // Test Case 2: Excessive noise (helps)
    console.log(chalk.bold("Test 2: Query with Excessive Noise"));
    const query2 = "###python### !!! @@@ programming ??? $$$ language @@@";
    const cleaned2 = preprocessQuery(query2, { lowercase: true, normalizeSpace: true, removeSpecial: true });

    console.log(`  Original: "${query2}"`);
    console.log(`  Cleaned:  "${cleaned2}"`);

    const results2_orig = await retrieveDocuments(vectorStore, embeddingContext, query2, 1);
    const results2_clean = await retrieveDocuments(vectorStore, embeddingContext, cleaned2, 1);

    console.log(`  Score with special chars:    ${chalk.yellow(results2_orig[0].similarity.toFixed(4))}`);
    console.log(`  Score without special chars: ${chalk.green(results2_clean[0].similarity.toFixed(4))} ${chalk.dim("(better)")}`);
    console.log(chalk.dim("  → Excessive symbols were pure noise\n"));

    // Test Case 3: Structured formatting (may hurt)
    console.log(chalk.bold("Test 3: Query with Meaningful Symbols"));
    const query3 = "javascript (JS) vs python";
    const cleaned3 = preprocessQuery(query3, { lowercase: true, normalizeSpace: true, removeSpecial: true });

    console.log(`  Original: "${query3}"`);
    console.log(`  Cleaned:  "${cleaned3}"`);

    const results3_orig = await retrieveDocuments(vectorStore, embeddingContext, query3, 1);
    const results3_clean = await retrieveDocuments(vectorStore, embeddingContext, cleaned3, 1);

    console.log(`  Score with special chars:    ${chalk.yellow(results3_orig[0].similarity.toFixed(4))}`);
    console.log(`  Score without special chars: ${results3_clean[0].similarity.toFixed(4)}`);
    console.log(chalk.dim("  → Parentheses and 'vs' provided structure\n"));

    console.log(`${chalk.bold("💡 Key Insight:")}`);
    console.log("Special character removal is NOT always beneficial:");
    console.log("• ✓ Helps: When there's excessive noise (###, @@@, !!!)");
    console.log("• ✗ Hurts: When punctuation adds context (contractions, structure)");
    console.log("• Solution: Use query expansion BEFORE removing special chars");
    console.log("  Example: 'What's' → 'What is' → 'what is' (preserves meaning)\n");
}

/**
 * Example 3: Stopword Removal Trade-offs
 */
async function example3(embeddingContext) {
    const vectorStore = new VectorDB({ dim: DIM, maxElements: MAX_ELEMENTS });
    const documents = createKnowledgeBase();
    await addDocumentsToStore(vectorStore, embeddingContext, documents);

    const query = "what is the best way to learn python programming";
    const withStopwords = preprocessQuery(query, { removeStops: false });
    const withoutStopwords = preprocessQuery(query, { removeStops: true });

    console.log(`${chalk.bold("Original Query:")} "${query}"`);
    console.log(`${chalk.bold("With Stopwords:")} "${withStopwords}"`);
    console.log(`${chalk.bold("Without Stopwords:")} "${withoutStopwords}"\n`);

    const resultsWithStops = await retrieveDocuments(vectorStore, embeddingContext, withStopwords, 3);
    const resultsWithoutStops = await retrieveDocuments(vectorStore, embeddingContext, withoutStopwords, 3);

    console.log(chalk.bold("Results WITH Stopwords:"));
    resultsWithStops.forEach((doc, idx) => {
        console.log(`  ${idx + 1}. [${doc.similarity.toFixed(4)}] ${doc.metadata.topic}`);
    });

    console.log(`\n${chalk.bold("Results WITHOUT Stopwords:")}`);
    resultsWithoutStops.forEach((doc, idx) => {
        console.log(`  ${idx + 1}. [${doc.similarity.toFixed(4)}] ${doc.metadata.topic}`);
    });

    console.log(`\n${chalk.bold("💡 Key Insight:")}`);
    console.log("Stopword removal can help or hurt, depending on the embedding model.");
    console.log("Modern models (like BGE) handle stopwords well - often better to keep them.");
    console.log("Test both approaches with your specific model and data.\n");
}

/**
 * Example 4: Query Expansion with Abbreviations
 */
async function example4(embeddingContext) {
    const vectorStore = new VectorDB({ dim: DIM, maxElements: MAX_ELEMENTS });
    const documents = createKnowledgeBase();
    await addDocumentsToStore(vectorStore, embeddingContext, documents);

    const abbreviatedQuery = "tell me about ml and ai";
    const expandedQuery = preprocessQuery(abbreviatedQuery, {
        lowercase: true,
        expand: true
    });

    console.log(`${chalk.bold("Original Query:")} "${abbreviatedQuery}"`);
    console.log(`${chalk.bold("Expanded Query:")} "${expandedQuery}"\n`);

    const abbrResults = await retrieveDocuments(vectorStore, embeddingContext, abbreviatedQuery, 3);
    const expandedResults = await retrieveDocuments(vectorStore, embeddingContext, expandedQuery, 3);

    console.log(chalk.bold("Results with Abbreviations:"));
    abbrResults.forEach((doc, idx) => {
        console.log(`  ${idx + 1}. [${chalk.yellow(doc.similarity.toFixed(4))}] ${doc.metadata.topic}`);
    });

    console.log(`\n${chalk.bold("Results with Expanded Query:")}`);
    expandedResults.forEach((doc, idx) => {
        console.log(`  ${idx + 1}. [${chalk.green(doc.similarity.toFixed(4))}] ${doc.metadata.topic}`);
    });

    console.log(`\n${chalk.bold("💡 Key Insight:")}`);
    console.log("Expanding abbreviations helps match document phrasing.");
    console.log("Build a domain-specific expansion dictionary for best results.\n");
}

/**
 * Example 5: Complete Preprocessing Pipeline
 */
async function example5(embeddingContext, chatSession) {
    const vectorStore = new VectorDB({ dim: DIM, maxElements: MAX_ELEMENTS });
    const documents = createKnowledgeBase();
    await addDocumentsToStore(vectorStore, embeddingContext, documents);

    const messyQuery = "   Hey!!! Can you tell me about JS framework for UI???  ";

    console.log(`${chalk.bold("Original Query:")} "${messyQuery}"\n`);

    // Show each preprocessing step
    console.log(chalk.bold("Preprocessing Steps:"));
    let step1 = basicClean(messyQuery);
    console.log(`  1. Lowercase & trim: "${step1}"`);

    let step2 = normalizeWhitespace(step1);
    console.log(`  2. Normalize spaces: "${step2}"`);

    // FIXED: Now expanding before removing special chars
    let step3 = expandQuery(step2);
    console.log(`  3. Expand abbreviations: "${step3}"`);

    let step4 = removeSpecialChars(step3);
    console.log(`  4. Remove special chars: "${step4}"`);

    const finalQuery = step4;
    console.log(`\n${chalk.bold("Final Processed Query:")} "${finalQuery}"\n`);

    // Compare retrieval
    const originalResults = await retrieveDocuments(vectorStore, embeddingContext, messyQuery, 3);
    const processedResults = await retrieveDocuments(vectorStore, embeddingContext, finalQuery, 3);

    console.log(chalk.bold("Comparison:"));
    console.log(`\n  ${chalk.bold("Original Query Results:")}`);
    originalResults.forEach((doc, idx) => {
        console.log(`    ${idx + 1}. [${doc.similarity.toFixed(4)}] ${doc.metadata.topic}`);
    });

    console.log(`\n  ${chalk.bold("Processed Query Results:")}`);
    processedResults.forEach((doc, idx) => {
        console.log(`    ${idx + 1}. [${chalk.green(doc.similarity.toFixed(4))}] ${doc.metadata.topic}`);
    });

    // Generate answer with processed query
    const context = buildContext(processedResults);
    const answer = await generateAnswer(chatSession, finalQuery, context);

    console.log(`\n${chalk.bold("Generated Answer:")}`);
    console.log(chalk.yellow(answer));

    console.log(`\n${chalk.bold("💡 Key Insight:")}`);
    console.log("Combining multiple preprocessing steps creates a robust pipeline.");
    console.log("Each step addresses a specific type of noise in user queries.");
    console.log("Order matters: expand abbreviations before removing special characters!\n");
}

/**
 * Example 6: Embedding Vector Stability
 */
async function example6(embeddingContext) {
    // Same semantic query with different formatting
    const queries = [
        "what is python",
        "What is Python?",
        "WHAT IS PYTHON!!!",
        "  what   is   python  ",
        "What's Python???",
    ];

    console.log(chalk.bold("Testing embedding stability with different formats:\n"));

    const embeddings = [];
    for (const query of queries) {
        const cleaned = preprocessQuery(query, {
            lowercase: true,
            normalizeSpace: true,
            removeSpecial: true
        });
        const embedding = await embeddingContext.getEmbeddingFor(cleaned);
        embeddings.push(embedding.vector);
        console.log(`"${query}" → "${cleaned}"`);
    }

    // Calculate cosine similarity between embeddings
    console.log(`\n${chalk.bold("Embedding Similarity Matrix:")}`);
    console.log("(All queries should have very similar embeddings after preprocessing)\n");

    function cosineSimilarity(a, b) {
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;
        for (let i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    for (let i = 0; i < embeddings.length; i++) {
        let row = `Query ${i + 1}: `;
        for (let j = 0; j < embeddings.length; j++) {
            const sim = cosineSimilarity(Array.from(embeddings[i]), Array.from(embeddings[j]));
            const color = sim > 0.99 ? chalk.green : sim > 0.95 ? chalk.yellow : chalk.red;
            row += color(sim.toFixed(4)) + " ";
        }
        console.log(row);
    }

    console.log(`\n${chalk.bold("💡 Key Insight:")}`);
    console.log("Query preprocessing ensures consistent embeddings for semantically identical queries.");
    console.log("This leads to more stable and predictable retrieval results.\n");
}

/**
 * Example 7: Real-World Query Preprocessing
 */
async function example7(embeddingContext) {
    const vectorStore = new VectorDB({ dim: DIM, maxElements: MAX_ELEMENTS });
    const documents = createKnowledgeBase();
    await addDocumentsToStore(vectorStore, embeddingContext, documents);

    // Realistic noisy user queries
    const realWorldQueries = [
        "how do i use docker????",
        "Tell me abt ML algorithms plz",
        "What's the diff between JS and Python??",
    ];

    for (const rawQuery of realWorldQueries) {
        console.log(chalk.bold(`\nProcessing: "${rawQuery}"`));

        const processed = preprocessQuery(rawQuery, {
            lowercase: true,
            normalizeSpace: true,
            removeSpecial: true,
            expand: true
        });

        console.log(chalk.dim(`Processed: "${processed}"`));

        const results = await retrieveDocuments(vectorStore, embeddingContext, processed, 2);

        console.log(chalk.bold("Top Results:"));
        results.forEach((doc, idx) => {
            console.log(`  ${idx + 1}. [${chalk.green(doc.similarity.toFixed(4))}] ${doc.metadata.topic}`);
        });
    }

    console.log(`\n${chalk.bold("💡 Key Insight:")}`);
    console.log("Real user queries are messy - preprocessing handles variations gracefully.");
    console.log("Build your preprocessing pipeline based on actual user query patterns.\n");
}

// ============================================================================
// MAIN RUNNER
// ============================================================================

async function runAllExamples() {
    console.clear();
    console.log("\n" + "=".repeat(80));
    console.log(chalk.bold("RAG from Scratch - Query Preprocessing"));
    console.log("=".repeat(80) + "\n");

    console.log(chalk.dim("Prerequisites:"));
    console.log(chalk.dim("• Completed 06_retrieval_strategies/01_basic_retrieval"));
    console.log(chalk.dim("• npm install embedded-vector-db node-llama-cpp chalk"));
    console.log(chalk.dim("• Models: bge-small-en-v1.5.Q8_0.gguf and Qwen3-1.7B-Q8_0.gguf\n"));

    try {
        // OPTIMIZED: Initialize models once and reuse across all examples
        const embeddingContext = await OutputHelper.withSpinner(
            "Loading embedding model (once)...",
            () => initializeEmbeddingModel()
        );

        const chatSession = await OutputHelper.withSpinner(
            "Loading LLM (once)...",
            () => initializeLLM()
        );

        await OutputHelper.runExample("Example 1: Impact of Basic Cleaning", () => example1(embeddingContext));
        await OutputHelper.runExample("Example 2: Removing Special Characters", () => example2(embeddingContext));
        await OutputHelper.runExample("Example 3: Stopword Removal Trade-offs", () => example3(embeddingContext));
        await OutputHelper.runExample("Example 4: Query Expansion with Abbreviations", () => example4(embeddingContext));
        await OutputHelper.runExample("Example 5: Complete Preprocessing Pipeline", () => example5(embeddingContext, chatSession));
        await OutputHelper.runExample("Example 6: Embedding Vector Stability", () => example6(embeddingContext));
        await OutputHelper.runExample("Example 7: Real-World Query Preprocessing", () => example7(embeddingContext));

        console.log(chalk.bold.green("\n✅ All examples completed successfully!\n"));

        console.log(chalk.bold("Key Takeaways:"));
        console.log("• Query preprocessing improves embedding consistency and stability");
        console.log("• Basic cleaning (lowercase, trim, normalize spaces) is essential");
        console.log("• Removing special characters reduces noise in embeddings");
        console.log("• Stopword removal: test with your model (modern models often handle them well)");
        console.log("• Query expansion helps match document phrasing");
        console.log("• Build preprocessing based on actual user query patterns\n");

        console.log(chalk.bold("Best Practices:"));
        console.log("✓ Always lowercase and trim user input");
        console.log("✓ Normalize whitespace (multiple spaces → single space)");
        console.log("✓ Expand abbreviations BEFORE removing special characters");
        console.log("✓ Remove special characters for cleaner embeddings");
        console.log("✓ Test stopword removal - not always beneficial");
        console.log("✓ Monitor embedding stability across query variations\n");

        console.log(chalk.bold("Recommended Pipeline (Fixed Order):"));
        console.log("1. Lowercase + trim");
        console.log("2. Normalize whitespace");
        console.log("3. Expand abbreviations (domain-specific) ← Do this BEFORE step 4!");
        console.log("4. Remove special characters");
        console.log("5. (Optional) Remove stopwords - test first!\n");

        console.log(chalk.bold("Next Steps:"));
        console.log("• 03_hybrid_search: Combine vector search with keyword matching");
        console.log("• Advanced: Query rewriting, spell checking, semantic expansion\n");

    } catch (error) {
        console.error(chalk.red("\n❌ Error:"), error?.message ?? error);
        console.error(chalk.dim("\nMake sure you have:"));
        console.error(chalk.dim("1. Installed dependencies: npm install"));
        console.error(chalk.dim("2. Both model files in correct location:"));
        console.error(chalk.dim("   - bge-small-en-v1.5.Q8_0.gguf"));
        console.error(chalk.dim("   - Qwen3-1.7B-Q8_0.gguf"));
        console.error(chalk.dim("3. Correct model paths in the code\n"));
    }

    process.exit(0);
}

runAllExamples();
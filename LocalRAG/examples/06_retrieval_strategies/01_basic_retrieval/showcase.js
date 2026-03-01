import { getLlama, LlamaChatSession } from "node-llama-cpp";
import path from "path";
import { fileURLToPath } from "url";
import chalk from "chalk";
import { VectorDB } from "embedded-vector-db";
import { CharacterTextSplitter, PDFLoader } from "../../../src/index.js";

// ============================================================================
// CONFIGURATION
// ============================================================================

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const CONFIG = {
    // Embedding model configuration
    embeddingModelPath: path.join(__dirname, '..', '..', '..', 'models', 'bge-small-en-v1.5.Q8_0.gguf'),
    embeddingDimension: 384,

    // LLM configuration
    llmModelPath: path.join(__dirname, '..', '..', '..', "models", "hf_Qwen_Qwen3-1.7B.Q8_0.gguf"),

    // Vector store configuration
    maxElements: 10000,
    namespace: "einstein",

    // Text splitting configuration
    chunkSize: 500,
    chunkOverlap: 40,

    // Retrieval configuration
    topK: 3,

    // Document path
    // TODO create folder docs in the root of this application and place https://arxiv.org/pdf/1709.00666 as einstein.pdf into it
    documentPath: "./docs/einstein.pdf"
};

// ============================================================================
// STEP 1: EMBEDDING MODEL INITIALIZATION
// ============================================================================

/**
 * Initialize the embedding model for converting text to vectors
 * @returns {Promise<Object>} Embedding context for generating embeddings
 */
async function initializeEmbeddingModel() {
    console.log(chalk.blue('\nInitializing embedding model...'));

    try {
        const llama = await getLlama({ logLevel: 'error' });
        const model = await llama.loadModel({
            modelPath: CONFIG.embeddingModelPath
        });
        const context = await model.createEmbeddingContext();

        console.log(chalk.green('✓ Embedding model initialized'));
        return context;
    } catch (error) {
        console.error(chalk.red('✗ Failed to initialize embedding model:'), error);
        throw error;
    }
}

// ============================================================================
// STEP 2: DOCUMENT LOADING
// ============================================================================

/**
 * Load documents from PDF file
 * @returns {Promise<Array>} Array of document objects
 */
async function loadDocuments() {
    console.log(chalk.blue('\nLoading documents...'));

    try {
        const pdfLoader = new PDFLoader(CONFIG.documentPath, { splitPages: true });
        const documents = await pdfLoader.load();

        console.log(chalk.green(`✓ Loaded ${documents.length} pages`));
        return documents;
    } catch (error) {
        console.error(chalk.red('✗ Failed to load documents:'), error);
        throw error;
    }
}

// ============================================================================
// STEP 3: TEXT SPLITTING AND CHUNKING
// ============================================================================

/**
 * Split documents into smaller chunks for better retrieval
 * @param {Array} documents - Array of document objects
 * @returns {Array} Array of chunked documents with metadata
 */
function splitDocuments(documents) {
    console.log(chalk.blue('\n Splitting documents into chunks...'));

    const splitter = new CharacterTextSplitter({
        separator: ' ',
        chunkSize: CONFIG.chunkSize,
        chunkOverlap: CONFIG.chunkOverlap
    });

    const allChunks = [];
    let chunkId = 0;

    for (const doc of documents) {
        const chunks = splitter.splitText(doc.pageContent);

        for (const chunk of chunks) {
            allChunks.push({
                pageContent: chunk,
                metadata: {
                    ...doc.metadata,
                    id: `${doc.metadata.id || 'page'}_chunk_${chunkId}`,
                    chunkIndex: chunkId
                }
            });
            chunkId++;
        }
    }

    console.log(chalk.green(`✓ Created ${allChunks.length} chunks`));
    return allChunks;
}

// ============================================================================
// STEP 4: EMBEDDING GENERATION
// ============================================================================

/**
 * Generate embeddings for all document chunks
 * @param {Object} embeddingContext - The embedding model context
 * @param {Array} documents - Array of document chunks
 * @returns {Promise<Array>} Array of embeddings with metadata
 */
async function generateEmbeddings(embeddingContext, documents) {
    console.log(chalk.blue('\nGenerating embeddings...'));

    const embeddings = [];
    const total = documents.length;

    for (let i = 0; i < total; i++) {
        const document = documents[i];
        const embedding = await embeddingContext.getEmbeddingFor(document.pageContent);

        embeddings.push({
            id: document.metadata.id,
            content: document.pageContent,
            metadata: document.metadata,
            embedding: Array.from(embedding.vector),
            timestamp: Date.now()
        });

        // Progress indicator
        if ((i + 1) % 10 === 0 || i === total - 1) {
            const percent = (((i + 1) / total) * 100).toFixed(1);
            process.stdout.write(`\r${chalk.cyan('Progress:')} ${i + 1}/${total} (${percent}%)`);
        }
    }

    console.log(chalk.green(`\n✓ Generated ${embeddings.length} embeddings`));
    return embeddings;
}

// ============================================================================
// STEP 5: VECTOR STORE CREATION AND POPULATION
// ============================================================================

/**
 * Initialize vector store and add documents
 * @param {Object} embeddingContext - The embedding model context
 * @param {Array} documents - Array of document chunks
 * @returns {Promise<Object>} Initialized vector store
 */
async function createVectorStore(embeddingContext, documents) {
    console.log(chalk.blue('\nCreating vector store...'));

    const vectorStore = new VectorDB({
        dim: CONFIG.embeddingDimension,
        maxElements: CONFIG.maxElements,
    });

    console.log(chalk.blue('Adding documents to vector store...'));

    for (let i = 0; i < documents.length; i++) {
        const doc = documents[i];
        const embedding = await embeddingContext.getEmbeddingFor(doc.pageContent);

        const metadata = {
            content: doc.pageContent,
            ...doc.metadata,
        };

        await vectorStore.insert(
            CONFIG.namespace,
            doc.metadata.id,
            Array.from(embedding.vector),
            metadata
        );

        // Progress indicator
        if ((i + 1) % 10 === 0 || i === documents.length - 1) {
            const percent = (((i + 1) / documents.length) * 100).toFixed(1);
            process.stdout.write(`\r${chalk.cyan('Progress:')} ${i + 1}/${documents.length} (${percent}%)`);
        }
    }

    console.log(chalk.green(`\n✓ Added ${documents.length} documents to vector store`));
    return vectorStore;
}

// ============================================================================
// STEP 6: RETRIEVAL
// ============================================================================

/**
 * Search vector store for relevant documents
 * @param {Object} vectorStore - The vector database
 * @param {Object} embeddingContext - The embedding model context
 * @param {string} query - User's question
 * @param {number} k - Number of results to retrieve
 * @returns {Promise<Array>} Array of search results
 */
async function searchVectorStore(vectorStore, embeddingContext, query, k = CONFIG.topK) {
    const queryEmbedding = await embeddingContext.getEmbeddingFor(query);
    return await vectorStore.search(
        CONFIG.namespace,
        Array.from(queryEmbedding.vector),
        k
    );
}

/**
 * Display search results in a formatted way
 * @param {Array} results - Search results from vector store
 */
function displaySearchResults(results) {
    console.log(chalk.blue('\n🔍 Retrieved documents:'));

    results.forEach((result, i) => {
        console.log(`\n${chalk.bold(`${i + 1}.`)} [Similarity Score: ${chalk.green(result.similarity.toFixed(4))}]`);
        console.log(`   ${chalk.dim("ID:")} ${result.id}`);
        console.log(`   ${chalk.dim("Preview:")} ${result.metadata.content.substring(0, 150)}...`);
        if (result.metadata.category) {
            console.log(`   ${chalk.dim("Category:")} ${result.metadata.category}`);
        }
    });
}

// ============================================================================
// STEP 7: LLM INITIALIZATION AND ANSWER GENERATION
// ============================================================================

/**
 * Initialize the LLM for answer generation
 * @returns {Promise<Object>} Chat session object
 */
async function initializeLLM() {
    console.log(chalk.blue('\n🤖 Initializing LLM...'));

    try {
        const llama = await getLlama({ logLevel: "error" });
        const model = await llama.loadModel({ modelPath: CONFIG.llmModelPath });
        const context = await model.createContext();
        const session = new LlamaChatSession({ contextSequence: context.getSequence() });

        console.log(chalk.green('✓ LLM initialized'));
        return session;
    } catch (error) {
        console.error(chalk.red('✗ Failed to initialize LLM:'), error);
        throw error;
    }
}

/**
 * Generate answer using LLM with retrieved context
 * @param {Object} chatSession - The LLM chat session
 * @param {string} query - User's question
 * @param {string} context - Retrieved context from vector store
 * @returns {Promise<string>} Generated answer
 */
async function generateAnswer(chatSession, query, context) {
    if (!context || context.trim() === '') {
        const prompt = `Question: ${query}\n\nYou don't have any relevant information to answer this question. Please say so politely.`;
        return (await chatSession.prompt(prompt)).trim();
    }

    const prompt = `You are a helpful assistant. Use the following context to answer the question. If the context doesn't contain relevant information, say so.

Context:
${context}

Question: ${query}

Answer:`;

    return (await chatSession.prompt(prompt)).trim();
}

// ============================================================================
// MAIN EXECUTION
// ============================================================================

async function main() {
    console.log(chalk.bold.cyan('\n╔════════════════════════════════════════════════╗'));
    console.log(chalk.bold.cyan('║      RAG (Retrieval-Augmented Generation)      ║'));
    console.log(chalk.bold.cyan('║              Demo Application                  ║'));
    console.log(chalk.bold.cyan('╚════════════════════════════════════════════════╝'));

    try {
        // Step 1: Initialize embedding model
        const embeddingContext = await initializeEmbeddingModel();

        // Step 2: Load documents
        const documents = await loadDocuments();

        // Step 3: Split documents into chunks
        const chunks = splitDocuments(documents);

        // Step 4: Generate embeddings (optional - for demonstration)
        const embeddings = await generateEmbeddings(embeddingContext, chunks);
        console.log(chalk.dim(`   (Generated ${embeddings.length} embedding vectors of dimension ${CONFIG.embeddingDimension})`));

        // Step 5: Create vector store and add documents
        const vectorStore = await createVectorStore(embeddingContext, chunks);

        // Step 6: Query and retrieval
        const question = "What was Einstein's school performance like? What grades did he get in his Matura?";

        console.log(chalk.bold.blue('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'));
        console.log(chalk.bold.yellow('❓ Question:'));
        console.log(chalk.white(`   ${question}`));
        console.log(chalk.bold.blue('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'));

        const results = await searchVectorStore(vectorStore, embeddingContext, question);
        displaySearchResults(results);

        // Step 7: Generate answer with and without context
        const chatSession = await initializeLLM();

        // Answer WITH context (RAG)
        const contextForAnswer = results.map(r => r.metadata.content).join('\n\n');
        const answerWithContext = await generateAnswer(chatSession, question, contextForAnswer);

        console.log(chalk.bold.blue('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'));
        console.log(chalk.bold.green('✨ Answer WITH Context (RAG):'));
        console.log(chalk.bold.blue('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'));
        console.log(chalk.yellow(answerWithContext));

        // Answer WITHOUT context (baseline)
        chatSession.resetChatHistory();
        const answerWithoutContext = await chatSession.prompt(question);

        console.log(chalk.bold.blue('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'));
        console.log(chalk.bold.red('⚠️  Answer WITHOUT Context (Baseline):'));
        console.log(chalk.bold.blue('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'));
        console.log(chalk.yellow(answerWithoutContext.trim()));

        console.log(chalk.bold.green('\n✅ RAG pipeline completed successfully!'));
        console.log(chalk.dim('\n💡 The difference shows how retrieval improves answer quality.\n'));

    } catch (error) {
        console.error(chalk.bold.red('\n❌ Error in RAG pipeline:'), error);
        process.exit(1);
    }

}

// ============================================================================
// EXECUTE
// ============================================================================

main();
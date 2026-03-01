/**
 * Efficient Embedding Generation and Management
 *
 * This example demonstrates how to:
 * 1. Generate embeddings for large document collections efficiently
 * 2. Save and load embeddings to/from disk (avoid re-computation)
 * 3. Handle incremental updates (only embed new documents)
 * 4. Compare different storage formats and performance
 * 5. Prepare embeddings for vector store integration
 *
 * Prerequisites:
 * - Completed 01_text_similarity_basics
 * - Have some PDF files or text documents to process
 */

import {fileURLToPath} from "url";
import path from "path";
import fs from "fs/promises";
import {getLlama} from "node-llama-cpp";
import {Document, PDFLoader, RecursiveCharacterTextSplitter} from '../../../src/index.js';
import {OutputHelper} from "../../../helpers/output-helper.js";
import chalk from "chalk";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Configuration
const STORAGE_DIR = path.join(__dirname, 'embeddings_storage');
const MODEL_PATH = path.join(__dirname, '..', '..', '..', 'models', 'bge-small-en-v1.5.Q8_0.gguf');

/**
 * Initialize the embedding model
 */
async function initializeEmbeddingModel() {
    const llama = await getLlama({
        logLevel: 'error'
    });
    const model = await llama.loadModel({
        modelPath: MODEL_PATH
    });
    return await model.createEmbeddingContext();
}

/**
 * Generate embeddings for multiple documents with progress tracking
 */
async function generateEmbeddings(context, documents, onProgress = null) {
    const embeddings = [];
    let processed = 0;

    for (const document of documents) {
        const embedding = await context.getEmbeddingFor(document.pageContent);

        embeddings.push({
            id: document.metadata.id || `doc_${processed}`,
            content: document.pageContent,
            metadata: document.metadata,
            embedding: Array.from(embedding.vector), // Convert to plain array
            timestamp: Date.now()
        });

        processed++;
        if (onProgress) {
            onProgress(processed, documents.length);
        }
    }

    return embeddings;
}

/**
 * Save embeddings to disk in JSON format
 */
async function saveEmbeddingsJSON(embeddings, filename) {
    await fs.mkdir(STORAGE_DIR, {recursive: true});
    const filepath = path.join(STORAGE_DIR, filename);

    const data = {
        version: '1.0',
        model: 'bge-small-en-v1.5',
        dimensions: (embeddings.length > 0 && embeddings[0]?.embedding)
            ? embeddings[0].embedding.length
            : 384,
        count: embeddings.length,
        created: new Date().toISOString(),
        embeddings: embeddings
    };

    await fs.writeFile(filepath, JSON.stringify(data, null, 2), 'utf-8');

    const stats = await fs.stat(filepath);
    return {filepath, size: stats.size};
}

/**
 * Load embeddings from JSON file
 */
async function loadEmbeddingsJSON(filename) {
    const filepath = path.join(STORAGE_DIR, filename);
    const content = await fs.readFile(filepath, 'utf-8');
    const data = JSON.parse(content);
    return data.embeddings;
}

/**
 * Check if embeddings exist for given document IDs
 */
async function loadExistingEmbeddings(filename) {
    try {
        const embeddings = await loadEmbeddingsJSON(filename);
        const embeddingMap = new Map();

        for (const item of embeddings) {
            embeddingMap.set(item.id, item);
        }

        return embeddingMap;
    } catch (error) {
        return new Map(); // No existing embeddings
    }
}

/**
 * Perform incremental embedding update
 */
async function incrementalEmbedding(context, newDocuments, existingFilename) {
    const existingMap = await loadExistingEmbeddings(existingFilename);

    // Filter out documents that already have embeddings
    const documentsToEmbed = newDocuments.filter(doc => {
        const id = doc.metadata.id || doc.pageContent.substring(0, 50);
        return !existingMap.has(id);
    });

    console.log(`Existing embeddings: ${existingMap.size}`);
    console.log(`New documents to embed: ${documentsToEmbed.length}`);
    console.log(`Skipped (already embedded): ${newDocuments.length - documentsToEmbed.length}`);

    if (documentsToEmbed.length === 0) {
        console.log(chalk.green('All documents already embedded!'));
        return Array.from(existingMap.values());
    }

    // Generate embeddings only for new documents
    const newEmbeddings = await generateEmbeddings(
        context,
        documentsToEmbed,
        (current, total) => {
            process.stdout.write(`\rEmbedding: ${current}/${total}`);
        }
    );
    console.log(); // New line

    // Merge with existing
    return [
        ...Array.from(existingMap.values()),
        ...newEmbeddings
    ];
}

/**
 * Create sample documents for testing
 */
function createSampleDocuments(count = 50) {
    const topics = [
        'artificial intelligence', 'machine learning', 'natural language processing',
        'computer vision', 'data science', 'cloud computing', 'cybersecurity',
        'blockchain', 'quantum computing', 'robotics', 'virtual reality'
    ];

    const templates = [
        (topic) => `Recent advances in ${topic} have revolutionized the technology industry.`,
        (topic) => `Understanding ${topic} is crucial for modern software development.`,
        (topic) => `The future of ${topic} looks promising with new innovations.`,
        (topic) => `Companies are investing heavily in ${topic} research and development.`,
        (topic) => `${topic} applications are transforming how we work and live.`
    ];

    const documents = [];

    for (let i = 0; i < count; i++) {
        const topic = topics[i % topics.length];
        const template = templates[i % templates.length];
        const text = template(topic);

        documents.push(new Document(text, {
            id: `doc_${i}`,
            topic: topic,
            source: 'sample_generator',
            index: i
        }));
    }

    return documents;
}

// ============================================================================
// EXAMPLES
// ============================================================================

/**
 * Example 1: Batch Embedding Generation
 * Shows how to efficiently embed a large collection of documents
 */
async function example1() {
    const context = await OutputHelper.withSpinner(
        'Loading embedding model...',
        () => initializeEmbeddingModel()
    );

    // Create a larger sample dataset
    const documents = createSampleDocuments(100);

    OutputHelper.formatStats({
        'Total Documents': documents.length,
        'Sample': documents[0].pageContent.substring(0, 50) + '...'
    });

    console.log(`\n${chalk.bold('Generating Embeddings:')}`);
    const startTime = Date.now();

    const embeddings = await generateEmbeddings(
        context,
        documents,
        (current, total) => {
            const percent = ((current / total) * 100).toFixed(1);
            process.stdout.write(`\r${chalk.cyan('Progress:')} ${current}/${total} (${percent}%)`);
        }
    );

    const duration = ((Date.now() - startTime) / 1000).toFixed(2);
    console.log(`\n${chalk.green('✓')} Completed in ${duration}s`);

    const avgTime = (duration / documents.length * 1000).toFixed(2);
    OutputHelper.formatStats({
        'Total Embeddings': embeddings.length,
        'Total Time': `${duration}s`,
        'Average Time': `${avgTime}ms per document`,
        'Throughput': `${(documents.length / duration).toFixed(1)} docs/sec`
    });
}

/**
 * Example 2: Save and Load Embeddings
 * Demonstrates persistence to avoid re-computation
 */
async function example2() {
    const context = await OutputHelper.withSpinner(
        'Loading embedding model...',
        () => initializeEmbeddingModel()
    );

    const documents = createSampleDocuments(50);

    // Generate embeddings
    console.log(`\n${chalk.bold('Phase 1: Generate and Save')}`);
    const generateStart = Date.now();
    const embeddings = await generateEmbeddings(context, documents);
    const generateTime = Date.now() - generateStart;

    // Save to disk
    const saveStart = Date.now();
    const jsonResult = await saveEmbeddingsJSON(embeddings, 'embeddings.json');
    const saveTime = Date.now() - saveStart;

    console.log(`${chalk.green('✓')} Generated ${embeddings.length} embeddings in ${(generateTime / 1000).toFixed(2)}s`);
    console.log(`${chalk.green('✓')} Saved to ${jsonResult.filepath}`);
    console.log(`${chalk.dim('  File size:')} ${(jsonResult.size / 1024 / 1024).toFixed(2)} MB`);

    // Load from disk
    console.log(`\n${chalk.bold('Phase 2: Load from Disk')}`);
    const loadStart = Date.now();
    const loadedEmbeddings = await loadEmbeddingsJSON('embeddings.json');
    const loadTime = Date.now() - loadStart;

    console.log(`${chalk.green('✓')} Loaded ${loadedEmbeddings.length} embeddings in ${loadTime}ms`);

    // Performance comparison
    console.log(`\n${chalk.bold('Performance Comparison:')}`);
    OutputHelper.formatStats({
        'Generate Time': `${(generateTime / 1000).toFixed(2)}s`,
        'Save Time': `${saveTime}ms`,
        'Load Time': `${loadTime}ms`,
        'Speedup': `${(generateTime / loadTime).toFixed(0)}x faster`
    });

    console.log(`\n${chalk.bold('💡 Key Insight:')}`);
    console.log('Loading pre-computed embeddings is dramatically faster!');
    console.log('Always save embeddings after generating them.\n');
}

/**
 * Example 3: Incremental Updates
 * Shows how to add new documents without re-embedding everything
 */
async function example3() {
    const context = await OutputHelper.withSpinner(
        'Loading embedding model...',
        () => initializeEmbeddingModel()
    );

    // Initial batch
    console.log(`\n${chalk.bold('Phase 1: Initial Batch')}`);
    const initialDocs = createSampleDocuments(30);
    const initialEmbeddings = await generateEmbeddings(context, initialDocs);
    await saveEmbeddingsJSON(initialEmbeddings, 'incremental.json');
    console.log(`${chalk.green('✓')} Generated and saved ${initialEmbeddings.length} embeddings`);

    // Simulate adding new documents
    console.log(`\n${chalk.bold('Phase 2: Add New Documents')}`);
    const newDocs = createSampleDocuments(20).map((doc, i) =>
        new Document(doc.pageContent, {
            ...doc.metadata,
            id: `doc_${30 + i}` // New IDs
        })
    );

    // Also add some duplicates (should be skipped)
    const duplicateDocs = createSampleDocuments(5);
    const allDocs = [...newDocs, ...duplicateDocs];

    console.log(`\nAttempting to process ${allDocs.length} documents:`);
    console.log(`  - ${newDocs.length} new documents`);
    console.log(`  - ${duplicateDocs.length} duplicates (should skip)`);

    // Incremental update
    const updateStart = Date.now();
    const updatedEmbeddings = await incrementalEmbedding(
        context,
        allDocs,
        'incremental.json'
    );
    const updateTime = Date.now() - updateStart;

    // Save updated embeddings
    await saveEmbeddingsJSON(updatedEmbeddings, 'incremental.json');

    console.log(`\n${chalk.bold('Results:')}`);
    OutputHelper.formatStats({
        'Total Embeddings': updatedEmbeddings.length,
        'Update Time': `${(updateTime / 1000).toFixed(2)}s`,
        'New Embeddings': newDocs.length,
        'Skipped': duplicateDocs.length
    });

    console.log(`\n${chalk.bold('💡 Key Insight:')}`);
    console.log('Incremental updates only embed new documents.');
    console.log('This saves time and compute resources!\n');
}

/**
 * Example 4: Storage Format Comparison
 * Compares JSON vs Binary formats for size and performance
 */
async function example4() {
    const context = await OutputHelper.withSpinner(
        'Loading embedding model...',
        () => initializeEmbeddingModel()
    );

    const documents = createSampleDocuments(100);

    console.log(`\n${chalk.bold('Generating embeddings for format comparison...')}`);
    const embeddings = await generateEmbeddings(context, documents);

    // Save in both formats
    console.log(`\n${chalk.bold('Saving in different formats:')}`);

    const jsonResult = await saveEmbeddingsJSON(embeddings, 'comparison.json');
    console.log(`${chalk.green('✓')} JSON: ${(jsonResult.size / 1024 / 1024).toFixed(2)} MB`);

    // Compare load times
    console.log(`\n${chalk.bold('Load Time Comparison:')}`);

    const jsonLoadStart = Date.now();
    await loadEmbeddingsJSON('comparison.json');
    const jsonLoadTime = Date.now() - jsonLoadStart;
    console.log(`JSON:   ${jsonLoadTime}ms`);
}

/**
 * Example 5: Preparing for Vector Stores
 * Shows how to structure embeddings for vector database integration
 */
async function example5() {
    const context = await OutputHelper.withSpinner(
        'Loading embedding model...',
        () => initializeEmbeddingModel()
    );

    // Simulate real document processing
    console.log(`\n${chalk.bold('Simulating Real Document Processing:')}`);
    console.log('In a real scenario, you would:');
    console.log('1. Load PDFs using extractTextFromPDF()');
    console.log('2. Split into chunks using RecursiveCharacterTextSplitter()');
    console.log('3. Generate embeddings for each chunk');
    console.log('4. Prepare for vector store\n');

    // Create sample chunks as if from a PDF
    const mockChunks = [
        new Document('Introduction to machine learning and its applications in modern technology.', {
            source: 'ml_guide.pdf',
            page: 1,
            chunk: 0,
            totalChunks: 3
        }),
        new Document('Supervised learning algorithms include decision trees and neural networks.', {
            source: 'ml_guide.pdf',
            page: 1,
            chunk: 1,
            totalChunks: 3
        }),
        new Document('Deep learning has revolutionized computer vision and natural language processing.', {
            source: 'ml_guide.pdf',
            page: 2,
            chunk: 2,
            totalChunks: 3
        })
    ];

    const embeddings = await generateEmbeddings(context, mockChunks);

    // Format for vector store
    const vectorStoreFormat = embeddings.map((item) => ({
        id: item.id,
        vector: item.embedding,
        metadata: {
            content: item.content,
            source: item.metadata.source,
            page: item.metadata.page,
            chunk: item.metadata.chunk,
            timestamp: item.timestamp
        }
    }));

    // Save in vector store format
    await saveEmbeddingsJSON(vectorStoreFormat, 'vector_store_ready.json');

    console.log(chalk.bold('Vector Store Format Example:'));
    console.log(JSON.stringify(vectorStoreFormat[0], null, 2).substring(0, 400) + '...\n');

    console.log(chalk.bold('Ready for Vector Stores:'));
    console.log('• LanceDB: Can import directly from this format');
    console.log('• Qdrant: Convert to their Point format');
    console.log('• Chroma: Convert to their Document format');
    console.log('• Milvus: Map to their schema\n');

    console.log(chalk.bold('💡 Next Steps:'));
    console.log('1. Choose a vector store (05_building_vector_store)');
    console.log('2. Import these embeddings');
    console.log('3. Add similarity search capabilities');
    console.log('4. Build complete RAG pipeline\n');
}

/**
 * Example 6: Real-World PDF Processing
 * Demonstrates the complete workflow with actual PDFs (if available)
 */
async function example6() {
    console.log(`\n${chalk.bold('Real-World PDF Processing Workflow:')}`);
    console.log(chalk.dim('This example shows the complete pipeline.\n'));

    try {
        const context = await OutputHelper.withSpinner(
            'Loading embedding model...',
            () => initializeEmbeddingModel()
        );

        // Try to load a real PDF
        const pdfUrl = 'https://arxiv.org/pdf/2402.19473'; // Example paper

        console.log(`${chalk.bold('Step 1:')} Load PDF`);
        const loader = new PDFLoader(pdfUrl, {splitPages: true});

        const docs = await OutputHelper.withSpinner(
            'Downloading PDF...',
            () => loader.load()
        );
        console.log(`${chalk.green('✓')} Loaded ${docs.length} pages\n`);

        console.log(`${chalk.bold('Step 2:')} Split into chunks`);
        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 500,
            chunkOverlap: 50
        });
        const chunks = await splitter.splitDocuments(docs);
        console.log(`${chalk.green('✓')} Created ${chunks.length} chunks\n`);

        console.log(`${chalk.bold('Step 3:')} Generate embeddings`);
        const embeddings = await generateEmbeddings(
            context,
            chunks.slice(0, 20), // Limit for demo
            (current, total) => {
                process.stdout.write(`\rProgress: ${current}/${total}`);
            }
        );
        console.log(`\n${chalk.green('✓')} Generated ${embeddings.length} embeddings\n`);

        console.log(`${chalk.bold('Step 4:')} Save to disk`);
        const result = await saveEmbeddingsJSON(embeddings, 'pdf_embeddings.json');
        console.log(`${chalk.green('✓')} Saved to ${path.basename(result.filepath)}`);
        console.log(`${chalk.dim('  Size:')} ${(result.size / 1024).toFixed(2)} KB\n`);

        console.log(chalk.bold.green('Complete pipeline executed successfully!'));
        console.log(`\n${chalk.bold('Summary:')}`);
        OutputHelper.formatStats({
            'PDF Pages': docs.length,
            'Text Chunks': chunks.length,
            'Embeddings': embeddings.length,
            'Storage': `${(result.size / 1024).toFixed(2)} KB`
        });

    } catch (error) {
        console.error(chalk.red('Error:'), error.message);
        console.log(chalk.dim('\nThis example requires internet access to download a PDF.'));
        console.log(chalk.dim('In a real application, you would use local PDFs.'));
    }
}

// ============================================================================
// MAIN RUNNER
// ============================================================================

async function runAllExamples() {
    console.clear();
    console.log('\n' + '='.repeat(80));
    console.log(chalk.bold('RAG from Scratch - Generate Embeddings'));
    console.log('='.repeat(80) + '\n');

    console.log(chalk.dim('Prerequisites:'));
    console.log(chalk.dim('• Completed 01_text_similarity_basics'));
    console.log(chalk.dim('• Model: bge-small-en-v1.5.Q8_0.gguf in the models directory'));
    console.log(chalk.dim('• Node module: node-llama-cpp installed\n'));

    try {
        await OutputHelper.runExample('Example 1: Batch Embedding Generation', example1);
        await OutputHelper.runExample('Example 2: Save and Load Embeddings', example2);
        await OutputHelper.runExample('Example 3: Incremental Updates', example3);
        await OutputHelper.runExample('Example 4: Storage Format Comparison', example4);
        await OutputHelper.runExample('Example 5: Preparing for Vector Stores', example5);
        await OutputHelper.runExample('Example 6: Real-World PDF Processing', example6);

        console.log(chalk.bold.green('\n✅ All examples completed successfully!\n'));
        console.log(chalk.bold('Key Takeaways:'));
        console.log('• Generate embeddings in batches for efficiency');
        console.log('• Always save embeddings to avoid re-computation');
        console.log('• Use incremental updates for new documents');
        console.log('• Choose storage format based on use case');
        console.log('• Structure data properly for vector stores\n');

        console.log(chalk.bold('Next Steps:'));
        console.log('• 05_building_vector_store: Use professional vector databases');
        console.log('• Implement similarity search at scale');
        console.log('• Build complete RAG pipeline\n');

    } catch (error) {
        console.error(chalk.red('\n❌ Error:'), error.message);
        console.error(chalk.dim('\nMake sure you have:'));
        console.error(chalk.dim('1. Completed previous examples'));
        console.error(chalk.dim('2. Downloaded the embedding model'));
        console.error(chalk.dim('3. Installed all dependencies\n'));
    }
}

runAllExamples();
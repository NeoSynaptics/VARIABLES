/**
 * 01_text_similarity_basics - Introduction to Text Similarity with Embeddings
 *
 * This example demonstrates the fundamental concepts of text embeddings and similarity search.
 * It shows how to:
 * 1. Convert text into numerical vectors (embeddings)
 * 2. Calculate similarity between texts using cosine similarity
 * 3. Find the most relevant documents for a given query
 *
 * Download the used model here: https://huggingface.co/ChristianAzinn/bge-small-en-v1.5-gguf/blob/main/bge-small-en-v1.5.Q8_0.gguf
 */

import {fileURLToPath} from "url";
import path from "path";
import {getLlama} from "node-llama-cpp";
import {Document} from '../../../src/index.js';
import {OutputHelper} from "../../../helpers/output-helper.js";
import chalk from "chalk";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Sample documents for our knowledge base
const sampleTexts = [
    "The sky is clear and blue today",
    "I love eating pizza with extra cheese",
    "Dogs love to play fetch with their owners",
    "The capital of France is Paris",
    "Drinking water is important for staying hydrated",
    "Mount Everest is the tallest mountain in the world",
    "A warm cup of tea is perfect for a cold winter day",
    "Painting is a form of creative expression",
    "Not all the things that shine are made of gold",
    "Cleaning the house is a good way to keep it tidy"
];

// Convert texts to Document objects for consistency with the rest of the tutorial
const documents = sampleTexts.map((text, i) =>
    new Document(text, { source: 'sample_data' })
);

/**
 * Initialize the embedding model
 * Note: You'll need to download an embedding model like bge-small-en-v1.5-Q8_0.gguf
 * and place it in the models directory
 */
async function initializeEmbeddingModel() {
    const llama = await getLlama({
        logLevel: 'error' // Only show errors, not warnings or info
    });
    const model = await llama.loadModel({
        modelPath: path.join(__dirname, '..', '..', '..', 'models', 'bge-small-en-v1.5.Q8_0.gguf')
    });
    return await model.createEmbeddingContext();
}

/**
 * Embed multiple documents in parallel for efficiency
 * Returns a Map where Document objects are keys and embeddings are values
 */
async function embedDocuments(context, documents) {
    const embeddings = new Map();

    await Promise.all(
        documents.map(async (document) => {
            const embedding = await context.getEmbeddingFor(document.pageContent);
            embeddings.set(document, embedding);
        })
    );

    return embeddings;
}

/**
 * Find documents most similar to a query embedding
 * Returns documents sorted by similarity (highest first)
 */
function findSimilarDocuments(queryEmbedding, documentEmbeddings, topK = 3) {
    const similarities = [];

    for (const [document, embedding] of documentEmbeddings) {
        const similarity = queryEmbedding.calculateCosineSimilarity(embedding);
        similarities.push({ document, similarity });
    }

    // Sort by similarity score (highest first)
    similarities.sort((a, b) => b.similarity - a.similarity);

    // Return top K results
    return similarities.slice(0, topK);
}

// EXAMPLES

/**
 * Example 1: Basic Text Similarity
 * Shows how embeddings capture semantic meaning, not just keywords
 */
async function example1() {
    const context = await OutputHelper.withSpinner(
        'Loading embedding model...',
        () => initializeEmbeddingModel()
    );

    OutputHelper.log.info(`Embedding ${documents.length} documents`);
    const documentEmbeddings = await OutputHelper.withSpinner(
        'Creating embeddings...',
        () => embedDocuments(context, documents)
    );

    const query = "What is the tallest mountain on Earth?";
    console.log(`\n${chalk.bold('Query:')} ${chalk.cyan(query)}`);

    const queryEmbedding = await context.getEmbeddingFor(query);
    const topResults = findSimilarDocuments(queryEmbedding, documentEmbeddings, 3);

    console.log(`\n${chalk.bold('Top 3 Most Similar Documents:')}`);
    console.log('-'.repeat(60));
    topResults.forEach((result, index) => {
        console.log(`${chalk.bold(`${index + 1}.`)} [Similarity: ${chalk.green(result.similarity.toFixed(4))}]`);
        console.log(`   "${chalk.dim(result.document.pageContent)}"\n`);
    });

    console.log(chalk.bold('💡 Key Insight:'));
    console.log('Notice how the top result is about Mount Everest, even though');
    console.log('the query and document use different words!');
    console.log('Embeddings capture MEANING, not just keyword matches.\n');
}

/**
 * Example 2: Multiple Queries
 * Demonstrates how the same embedded documents can answer different queries
 */
async function example2() {
    const context = await OutputHelper.withSpinner(
        'Loading embedding model...',
        () => initializeEmbeddingModel()
    );

    const documentEmbeddings = await OutputHelper.withSpinner(
        'Embedding documents...',
        () => embedDocuments(context, documents)
    );

    const queries = [
        "Tell me about hydration",
        "What's a good winter drink?",
        "Information about European capitals"
    ];

    for (const query of queries) {
        console.log(`\n${chalk.bold('Query:')} ${chalk.cyan(query)}`);
        const queryEmbedding = await context.getEmbeddingFor(query);
        const topResult = findSimilarDocuments(queryEmbedding, documentEmbeddings, 1)[0];

        console.log(`${chalk.bold('Best Match:')} [${chalk.green(topResult.similarity.toFixed(4))}]`);
        console.log(`"${chalk.dim(topResult.document.pageContent)}"`);
    }

    console.log(`\n${chalk.bold('💡 Efficiency:')}`);
    console.log('Documents are embedded ONCE, then reused for multiple queries.');
    console.log('This is the foundation of fast semantic search!\n');
}

/**
 * Example 3: Understanding Embedding Vectors
 * Shows what embeddings actually look like under the hood
 */
async function example3() {
    const context = await OutputHelper.withSpinner(
        'Loading embedding model...',
        () => initializeEmbeddingModel()
    );

    const sampleDoc = documents[0];
    const embedding = await context.getEmbeddingFor(sampleDoc.pageContent);

    OutputHelper.formatStats({
        'Document': `"${sampleDoc.pageContent}"`,
        'Vector Dimensions': embedding.vector.length,
        'Vector Type': 'Array of floating-point numbers',
        'Storage Size': `~${(embedding.vector.length * 4 / 1024).toFixed(2)} KB`
    });

    console.log(`\n${chalk.bold('Sample Vector Values (first 10):')}`);
    console.log(`[${embedding.vector.slice(0, 10).map(v => v.toFixed(4)).join(', ')}...]\n`);

    console.log(chalk.bold('💡 How It Works:'));
    console.log('• Each document becomes a vector in high-dimensional space');
    console.log('• Similar documents have similar vectors');
    console.log('• We use cosine similarity to measure "closeness"');
    console.log('• Cosine similarity ranges from -1 (opposite) to 1 (identical)\n');
}

/**
 * Example 4: Similarity Score Distribution
 * Analyzes how similarity scores are distributed across all documents
 */
async function example4() {
    const context = await OutputHelper.withSpinner(
        'Loading embedding model...',
        () => initializeEmbeddingModel()
    );

    const documentEmbeddings = await OutputHelper.withSpinner(
        'Embedding documents...',
        () => embedDocuments(context, documents)
    );

    const query = "Tell me about nature and weather";
    console.log(`\n${chalk.bold('Query:')} ${chalk.cyan(query)}`);

    const queryEmbedding = await context.getEmbeddingFor(query);
    const allResults = findSimilarDocuments(queryEmbedding, documentEmbeddings, documents.length);

    console.log(`\n${chalk.bold('All Documents Ranked by Similarity:')}`);
    console.log('-'.repeat(70));

    allResults.forEach((result, index) => {
        const score = result.similarity.toFixed(4);
        const bar = '█'.repeat(Math.round(result.similarity * 30));
        const color = result.similarity > 0.3 ? chalk.green :
            result.similarity > 0.2 ? chalk.yellow : chalk.gray;

        console.log(`${String(index + 1).padStart(2)}. ${color(score)} ${color(bar)}`);
        console.log(`    ${chalk.dim(result.document.pageContent)}\n`);
    });

    const scores = allResults.map(r => r.similarity);
    const avg = scores.reduce((a, b) => a + b, 0) / scores.length;

    console.log(chalk.bold('📊 Statistics:'));
    console.log(`  Average similarity: ${avg.toFixed(4)}`);
    console.log(`  Highest score: ${Math.max(...scores).toFixed(4)}`);
    console.log(`  Lowest score: ${Math.min(...scores).toFixed(4)}\n`);
}

/**
 * Example 5: Comparing Different Queries
 * Shows how query formulation affects results
 */
async function example5() {
    const context = await OutputHelper.withSpinner(
        'Loading embedding model...',
        () => initializeEmbeddingModel()
    );

    const documentEmbeddings = await OutputHelper.withSpinner(
        'Embedding documents...',
        () => embedDocuments(context, documents)
    );

    console.log('\n' + chalk.bold('Query Comparison Study:') + '\n');

    const queryPairs = [
        {
            type: 'Keyword vs Semantic',
            queries: [
                "mountain tallest",
                "What is the highest peak in the world?"
            ]
        },
        {
            type: 'Direct vs Contextual',
            queries: [
                "France capital",
                "Which city is the government center of France?"
            ]
        }
    ];

    for (const pair of queryPairs) {
        console.log(chalk.bold(`\n${pair.type}:`));
        console.log('-'.repeat(60));

        for (const query of pair.queries) {
            const queryEmbedding = await context.getEmbeddingFor(query);
            const topResult = findSimilarDocuments(queryEmbedding, documentEmbeddings, 1)[0];

            console.log(`\n${chalk.cyan('Query:')} "${query}"`);
            console.log(`${chalk.green('Result:')} "${topResult.document.pageContent}"`);
            console.log(`${chalk.yellow('Score:')} ${topResult.similarity.toFixed(4)}`);
        }
    }

    console.log(`\n${chalk.bold('💡 Takeaway:')}`);
    console.log('Both keyword-heavy and natural language queries work well!');
    console.log('Embeddings understand meaning regardless of phrasing.\n');
}

// ============================================================================
// MAIN RUNNER
// ============================================================================

async function runAllExamples() {
    console.clear();
    console.log('\n' + '='.repeat(80));
    console.log(chalk.bold('RAG from Scratch - Text Similarity Examples'));
    console.log('='.repeat(80) + '\n');

    console.log(chalk.dim('Prerequisites:'));
    console.log(chalk.dim('• npm install node-llama-cpp'));
    console.log(chalk.dim('• Download bge-small-en-v1.5-Q8_0.gguf model'));
    console.log(chalk.dim('• Place model in models/ folder in the root of this project\n'));

    try {
        await OutputHelper.runExample('Example 1: Basic Text Similarity', example1);
        await OutputHelper.runExample('Example 2: Multiple Queries', example2);
        await OutputHelper.runExample('Example 3: Understanding Embedding Vectors', example3);
        await OutputHelper.runExample('Example 4: Similarity Score Distribution', example4);
        await OutputHelper.runExample('Example 5: Comparing Different Queries', example5);

        console.log(chalk.bold.green('\n ✅ All examples completed successfully!\n'));
    } catch (error) {
        console.error(chalk.red('\n❌ Error:'), error.message);
        console.error(chalk.dim('\nMake sure you have:'));
        console.error(chalk.dim('1. Installed node-llama-cpp: npm install node-llama-cpp'));
        console.error(chalk.dim('2. Downloaded bge-small-en-v1.5-q8_0.gguf model into the models folder'));
        console.error(chalk.dim('3. Placed the model in the correct directory\n'));
    }
}

runAllExamples();
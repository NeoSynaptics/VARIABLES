/**
 * Hybrid Search for RAG - E-Commerce Focus
 *
 * Demonstrates:
 * 1) Product search combining semantic understanding with exact matches
 * 2) Score normalization techniques for fair combination
 * 3) Multi-field hybrid search (title, description, attributes)
 * 4) Dynamic weight adjustment based on query patterns
 * 5) Handling out-of-vocabulary (OOV) terms like SKUs and brands
 * 6) Performance optimization strategies
 *
 * Why This Approach?
 * - Real-world e-commerce has unique challenges
 * - Users mix natural language with product codes
 * - Brand names and SKUs need exact matching
 * - Semantic search helps with descriptive queries
 *
 * Prerequisites:
 * - npm install embedded-vector-db node-llama-cpp chalk
 * - Place bge-small-en-v1.5.Q8_0.gguf under models/
 */

import { fileURLToPath } from "url";
import path from "path";
import { VectorDB } from "embedded-vector-db";
import { getLlama } from "node-llama-cpp";
import { Document } from "../../../src/index.js";
import { OutputHelper } from "../../../helpers/output-helper.js";
import chalk from "chalk";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const EMBEDDING_MODEL_PATH = path.join(__dirname, "..", "..", "..", "models", "bge-small-en-v1.5.Q8_0.gguf");

// VectorDB config
const DIM = 384;
const MAX_ELEMENTS = 10000;
const NS = "product_search";

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
 * Create an e-commerce product catalog
 */
function createProductCatalog() {
    return [
        new Document("Apple MacBook Pro 16-inch with M3 Max chip, 32GB RAM, and 1TB SSD. Perfect for professional video editing, 3D rendering, and software development.", {
            id: "PROD-001",
            title: "MacBook Pro 16-inch M3 Max",
            brand: "Apple",
            category: "laptops",
            price: 3499,
            sku: "MBP-M3MAX-32-1TB",
            attributes: "M3 Max, 32GB RAM, 1TB SSD, 16-inch, Retina Display"
        }),
        new Document("Dell XPS 15 premium laptop featuring Intel i9 processor, 32GB DDR5 RAM, NVIDIA RTX 4060 graphics. Ideal for content creators and gamers.", {
            id: "PROD-002",
            title: "Dell XPS 15 Intel i9",
            brand: "Dell",
            category: "laptops",
            price: 2799,
            sku: "XPS15-I9-32-RTX4060",
            attributes: "Intel i9, 32GB RAM, RTX 4060, 15.6-inch, 4K OLED"
        }),
        new Document("Sony WH-1000XM5 wireless noise-cancelling headphones with industry-leading ANC technology. 30-hour battery life, premium sound quality.", {
            id: "PROD-003",
            title: "Sony WH-1000XM5 Headphones",
            brand: "Sony",
            category: "audio",
            price: 399,
            sku: "SONY-WH1000XM5-BLK",
            attributes: "Wireless, Noise-Cancelling, 30hr Battery, Bluetooth 5.2"
        }),
        new Document("Bose QuietComfort 45 over-ear headphones with world-class noise cancellation. Comfortable for all-day wear, excellent for travel and work.", {
            id: "PROD-004",
            title: "Bose QuietComfort 45",
            brand: "Bose",
            category: "audio",
            price: 329,
            sku: "BOSE-QC45-WHT",
            attributes: "Wireless, Noise-Cancelling, 24hr Battery, Lightweight"
        }),
        new Document("iPad Pro 12.9-inch with M2 chip and Apple Pencil support. ProMotion 120Hz display, perfect for digital artists and designers.", {
            id: "PROD-005",
            title: "iPad Pro 12.9-inch M2",
            brand: "Apple",
            category: "tablets",
            price: 1099,
            sku: "IPAD-PRO-M2-128",
            attributes: "M2 chip, 128GB, 12.9-inch, ProMotion, Apple Pencil"
        }),
        new Document("Samsung Galaxy Tab S9 Ultra with stunning AMOLED display and S Pen included. Powerful for productivity and entertainment.", {
            id: "PROD-006",
            title: "Samsung Galaxy Tab S9 Ultra",
            brand: "Samsung",
            category: "tablets",
            price: 1199,
            sku: "TAB-S9-ULTRA-256",
            attributes: "Snapdragon 8, 256GB, 14.6-inch, AMOLED, S Pen"
        }),
        new Document("Logitech MX Master 3S ergonomic wireless mouse with customizable buttons and ultra-precise scrolling. Best for productivity professionals.", {
            id: "PROD-007",
            title: "Logitech MX Master 3S Mouse",
            brand: "Logitech",
            category: "accessories",
            price: 99,
            sku: "LOG-MXMASTER3S-GRY",
            attributes: "Wireless, Ergonomic, 8000 DPI, Multi-Device"
        }),
        new Document("Apple Magic Keyboard with Touch ID for Mac. Wireless, rechargeable, with scissor mechanism for comfortable typing experience.", {
            id: "PROD-008",
            title: "Apple Magic Keyboard Touch ID",
            brand: "Apple",
            category: "accessories",
            price: 149,
            sku: "APPLE-MKEY-TOUCHID",
            attributes: "Wireless, Touch ID, Rechargeable, Scissor Keys"
        }),
        new Document("LG 27-inch 4K UHD monitor with IPS panel and HDR10 support. Color-accurate display perfect for photo and video editing.", {
            id: "PROD-009",
            title: "LG 27-inch 4K UHD Monitor",
            brand: "LG",
            category: "monitors",
            price: 449,
            sku: "LG-27UK-IPS-HDR",
            attributes: "27-inch, 4K, IPS, HDR10, 99% sRGB"
        }),
        new Document("Samsung Odyssey G7 32-inch curved gaming monitor with 240Hz refresh rate and 1ms response time. QLED technology for vibrant colors.", {
            id: "PROD-010",
            title: "Samsung Odyssey G7 Gaming Monitor",
            brand: "Samsung",
            category: "monitors",
            price: 699,
            sku: "SAM-G7-32-240HZ",
            attributes: "32-inch, 1440p, 240Hz, 1ms, Curved, QLED"
        }),
        new Document("Anker PowerCore 20000mAh portable charger with fast charging support. Compact design, charges multiple devices simultaneously.", {
            id: "PROD-011",
            title: "Anker PowerCore 20000mAh",
            brand: "Anker",
            category: "accessories",
            price: 49,
            sku: "ANK-PC20K-BLK",
            attributes: "20000mAh, Fast Charging, Dual USB, Portable"
        }),
        new Document("ASUS ROG Zephyrus G14 gaming laptop with AMD Ryzen 9 and RTX 4070. Compact 14-inch form factor, excellent for gaming on the go.", {
            id: "PROD-012",
            title: "ASUS ROG Zephyrus G14",
            brand: "ASUS",
            category: "laptops",
            price: 1899,
            sku: "ASUS-G14-R9-4070",
            attributes: "AMD Ryzen 9, RTX 4070, 16GB RAM, 1TB SSD, 14-inch"
        })
    ];
}

/**
 * Add products to vector store with multi-field indexing
 */
async function addProductsToStore(vectorStore, embeddingContext, products) {
    try {
        // Index multiple fields for comprehensive search
        await vectorStore.setFullTextIndexedFields(NS, ['content', 'title', 'brand', 'sku', 'attributes']);

        for (const product of products) {
            const embedding = await embeddingContext.getEmbeddingFor(product.pageContent);
            const metadata = {
                content: product.pageContent,
                ...product.metadata,
            };
            await vectorStore.insert(
                NS,
                product.metadata.id,
                Array.from(embedding.vector),
                metadata
            );
        }
    } catch (error) {
        throw new Error(`Failed to add products to store: ${error.message}`);
    }
}

/**
 * Example 1: SKU and Brand Name Challenges
 * Demonstrates why keyword search is critical for product codes
 */
async function example1(embeddingContext) {
    const vectorStore = new VectorDB({ dim: DIM, maxElements: MAX_ELEMENTS });
    const products = createProductCatalog();

    await OutputHelper.withSpinner(
        "Setting up product catalog...",
        () => addProductsToStore(vectorStore, embeddingContext, products)
    );

    // Test 1: Exact SKU search
    console.log(chalk.bold.cyan("Test 1: Exact SKU Search"));
    const query1 = "MBP-M3MAX-32-1TB";
    console.log(`${chalk.bold("Query:")} "${query1}"`);
    console.log(chalk.dim("Looking for exact product code\n"));

    const queryEmbedding1 = await embeddingContext.getEmbeddingFor(query1);
    const vectorResults1 = await vectorStore.search(NS, Array.from(queryEmbedding1.vector), 3);

    console.log(chalk.bold("Vector Search Results:"));
    vectorResults1.forEach((doc, idx) => {
        console.log(`  ${idx + 1}. [${chalk.yellow(doc.similarity.toFixed(4))}] ${doc.metadata.title}`);
        console.log(`     SKU: ${doc.metadata.sku}`);
    });

    const keywordResults1 = await vectorStore.fullTextSearch(NS, query1, 3);

    console.log(`\n${chalk.bold("Keyword Search Results:")}`);
    keywordResults1.forEach((doc, idx) => {
        console.log(`  ${idx + 1}. [${chalk.green(doc.similarity.toFixed(4))}] ${doc.metadata.title}`);
        console.log(`     SKU: ${doc.metadata.sku}`);
    });

    console.log(`\n${chalk.dim("Why keyword search dominates:")}`);
    console.log(chalk.dim('• SKUs are out-of-vocabulary (OOV) for embedding models'));
    console.log(chalk.dim('• Vector search treats "MBP-M3MAX" as random characters'));
    console.log(chalk.dim('• BM25 finds exact string match immediately'));
    console.log(chalk.dim('• Critical for product code searches!\n'));

    // Test 2: Brand-specific search
    console.log(chalk.bold.cyan("Test 2: Brand Name Search"));
    const query2 = "Sony headphones";
    console.log(`${chalk.bold("Query:")} "${query2}"`);
    console.log(chalk.dim("Brand name + category\n"));

    const queryEmbedding2 = await embeddingContext.getEmbeddingFor(query2);
    const vectorResults2 = await vectorStore.search(NS, Array.from(queryEmbedding2.vector), 3);

    console.log(chalk.bold("Vector Search Results:"));
    vectorResults2.forEach((doc, idx) => {
        console.log(`  ${idx + 1}. [${chalk.yellow(doc.similarity.toFixed(4))}] ${doc.metadata.brand} - ${doc.metadata.title}`);
    });

    const keywordResults2 = await vectorStore.fullTextSearch(NS, query2, 3);

    console.log(`\n${chalk.bold("Keyword Search Results:")}`);
    keywordResults2.forEach((doc, idx) => {
        console.log(`  ${idx + 1}. [${chalk.green(doc.similarity.toFixed(4))}] ${doc.metadata.brand} - ${doc.metadata.title}`);
    });

    console.log(`\n${chalk.dim("Observation:")}`);
    console.log(chalk.dim('• Both methods find Sony products'));
    console.log(chalk.dim('• Keyword search prioritizes exact brand match'));
    console.log(chalk.dim('• Vector search might include similar products (Bose)'));
    console.log(chalk.dim('• Hybrid gives precise brand + semantic category\n'));

    console.log(`${chalk.bold("💡 Key Insight:")}`);
    console.log("For e-commerce, keyword search is essential for:");
    console.log("• Product codes (SKUs, model numbers)");
    console.log("• Brand names (proper nouns)");
    console.log("• Technical specs (exact numbers, acronyms)");
    console.log("• Part numbers and identifiers\n");
}

/**
 * Example 2: Score Normalization Techniques
 * Shows different methods to normalize scores before combining
 */
async function example2(embeddingContext) {
    const vectorStore = new VectorDB({ dim: DIM, maxElements: MAX_ELEMENTS });
    const products = createProductCatalog();

    await addProductsToStore(vectorStore, embeddingContext, products);

    const query = "powerful laptop for video editing";
    console.log(`${chalk.bold("Query:")} "${query}"\n`);

    const queryEmbedding = await embeddingContext.getEmbeddingFor(query);

    // Get raw scores
    const vectorResults = await vectorStore.search(NS, Array.from(queryEmbedding.vector), 5);
    const keywordResults = await vectorStore.fullTextSearch(NS, query, 5);

    console.log(chalk.bold("Raw Score Ranges:"));
    const vecScores = vectorResults.map(r => r.similarity);
    const keyScores = keywordResults.map(r => r.similarity);
    console.log(`Vector:  [${chalk.yellow(Math.min(...vecScores).toFixed(3))} - ${Math.max(...vecScores).toFixed(3)}]`);
    console.log(`Keyword: [${chalk.green(Math.min(...keyScores).toFixed(3))} - ${Math.max(...keyScores).toFixed(3)}]`);
    console.log(chalk.dim("Problem: Different scales make direct combination unfair!\n"));

    // Method 1: Min-Max Normalization
    console.log(chalk.bold("Method 1: Min-Max Normalization"));
    console.log(chalk.dim("Formula: (score - min) / (max - min)"));
    console.log(chalk.dim("Range: [0, 1]\n"));

    const normalizeMinMax = (scores) => {
        const min = Math.min(...scores);
        const max = Math.max(...scores);
        const range = max - min || 1;
        return scores.map(s => (s - min) / range);
    };

    const vecNorm1 = normalizeMinMax(vecScores);
    const keyNorm1 = normalizeMinMax(keyScores);

    console.log("Normalized scores:");
    console.log(`Vector:  [${chalk.yellow(vecNorm1[0].toFixed(3))} ... ${vecNorm1[vecNorm1.length-1].toFixed(3)}]`);
    console.log(`Keyword: [${chalk.green(keyNorm1[0].toFixed(3))} ... ${keyNorm1[keyNorm1.length-1].toFixed(3)}]`);

    // Method 2: Z-Score Normalization
    console.log(`\n${chalk.bold("Method 2: Z-Score Normalization")}`);
    console.log(chalk.dim("Formula: (score - mean) / std_dev"));
    console.log(chalk.dim("Centers around 0, preserves outliers\n"));

    const normalizeZScore = (scores) => {
        const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
        const variance = scores.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / scores.length;
        const stdDev = Math.sqrt(variance) || 1;
        return scores.map(s => (s - mean) / stdDev);
    };

    const vecNorm2 = normalizeZScore(vecScores);
    const keyNorm2 = normalizeZScore(keyScores);

    console.log("Normalized scores:");
    console.log(`Vector:  [${chalk.yellow(vecNorm2[0].toFixed(3))} ... ${vecNorm2[vecNorm2.length-1].toFixed(3)}]`);
    console.log(`Keyword: [${chalk.green(keyNorm2[0].toFixed(3))} ... ${keyNorm2[keyNorm2.length-1].toFixed(3)}]`);

    // Method 3: Rank-based (Percentile)
    console.log(`\n${chalk.bold("Method 3: Rank-Based (Percentile)")}`);
    console.log(chalk.dim("Formula: rank / total_results"));
    console.log(chalk.dim("Simple, robust to score distribution\n"));

    const normalizeRank = (scores) => {
        const n = scores.length;
        return scores.map((_, idx) => (n - idx) / n);
    };

    const vecNorm3 = normalizeRank(vecScores);
    const keyNorm3 = normalizeRank(keyScores);

    console.log("Normalized scores:");
    console.log(`Vector:  [${chalk.yellow(vecNorm3[0].toFixed(3))} ... ${vecNorm3[vecNorm3.length-1].toFixed(3)}]`);
    console.log(`Keyword: [${chalk.green(keyNorm3[0].toFixed(3))} ... ${keyNorm3[keyNorm3.length-1].toFixed(3)}]`);

    console.log(`\n${chalk.bold("💡 Key Insight:")}`);
    console.log("Normalization Comparison:");
    console.log("• Min-Max: Simple, but sensitive to outliers");
    console.log("• Z-Score: Preserves distribution, good for statistical analysis");
    console.log("• Rank-Based: Most robust, used by RRF");
    console.log("• Choice depends on score distribution and use case\n");
}

/**
 * Example 3: Multi-Field Hybrid Search
 * Demonstrates searching across multiple product fields
 */
async function example3(embeddingContext) {
    const vectorStore = new VectorDB({ dim: DIM, maxElements: MAX_ELEMENTS });
    const products = createProductCatalog();

    await addProductsToStore(vectorStore, embeddingContext, products);

    const query = "Apple wireless keyboard";
    console.log(`${chalk.bold("Query:")} "${query}"`);
    console.log(chalk.dim("Should match brand field 'Apple' + product type 'keyboard'\n"));

    const queryEmbedding = await embeddingContext.getEmbeddingFor(query);

    // Single-field search (content only)
    console.log(chalk.bold("Standard Search (content field only):"));
    const standardResults = await vectorStore.hybridSearch(
        NS,
        Array.from(queryEmbedding.vector),
        query,
        { k: 3 }
    );

    standardResults.forEach((doc, idx) => {
        console.log(`  ${idx + 1}. [${chalk.cyan(doc.combinedScore.toFixed(4))}] ${doc.metadata.title}`);
        console.log(`     Brand: ${doc.metadata.brand} | Category: ${doc.metadata.category}`);
    });

    console.log(`\n${chalk.dim("Analysis:")}`);
    console.log(chalk.dim('• Searches across: content, title, brand, sku, attributes'));
    console.log(chalk.dim('• "Apple" matches in brand field strongly'));
    console.log(chalk.dim('• "wireless keyboard" matches in title and attributes'));
    console.log(chalk.dim('• Multi-field indexing captures all relevant signals\n'));

    // Compare with another query
    const query2 = "monitor for photo editing color accurate";
    console.log(`${chalk.bold("Query:")} "${query2}"\n`);

    const queryEmbedding2 = await embeddingContext.getEmbeddingFor(query2);
    const results2 = await vectorStore.hybridSearch(
        NS,
        Array.from(queryEmbedding2.vector),
        query2,
        { k: 3 }
    );

    console.log(chalk.bold("Results:"));
    results2.forEach((doc, idx) => {
        console.log(`  ${idx + 1}. [${chalk.cyan(doc.combinedScore.toFixed(4))}] ${doc.metadata.title}`);
        console.log(`     ${chalk.dim(doc.metadata.attributes)}`);
    });

    console.log(`\n${chalk.bold("💡 Key Insight:")}`);
    console.log("Multi-field indexing allows:");
    console.log("• Brand matching in brand field (exact match)");
    console.log("• Spec matching in attributes field (technical terms)");
    console.log("• Description matching in content field (semantic)");
    console.log("• SKU matching in sku field (product codes)");
    console.log("→ Comprehensive search across all product dimensions\n");
}

/**
 * Example 4: Query Pattern Detection
 * Automatically adjusts weights based on query characteristics
 */
async function example4(embeddingContext) {
    const vectorStore = new VectorDB({ dim: DIM, maxElements: MAX_ELEMENTS });
    const products = createProductCatalog();

    await addProductsToStore(vectorStore, embeddingContext, products);

    /**
     * Analyze query to determine optimal search strategy
     */
    function analyzeQuery(query) {
        const upperCount = (query.match(/[A-Z]/g) || []).length;
        const digitCount = (query.match(/\d/g) || []).length;
        const hyphenCount = (query.match(/-/g) || []).length;
        const wordCount = query.split(/\s+/).length;
        const hasQuestionWord = /^(what|how|which|where|why|when|who)/i.test(query);

        // Detect query type
        let queryType, weights;

        if (hyphenCount >= 2 || (upperCount > 3 && digitCount > 0)) {
            queryType = "SKU/Model Number";
            weights = { vector: 0.2, text: 0.8 };
        } else if (digitCount >= 3) {
            queryType = "Technical Specs";
            weights = { vector: 0.3, text: 0.7 };
        } else if (hasQuestionWord || wordCount >= 6) {
            queryType = "Natural Language Question";
            weights = { vector: 0.8, text: 0.2 };
        } else if (wordCount <= 2) {
            queryType = "Short Keyword";
            weights = { vector: 0.4, text: 0.6 };
        } else {
            queryType = "Mixed Query";
            weights = { vector: 0.5, text: 0.5 };
        }

        return { queryType, weights };
    }

    const testQueries = [
        "ASUS-G14-R9-4070",
        "32GB RAM 4K display",
        "What's the best laptop for video editing under 2000?",
        "Sony headphones",
        "portable charger fast charging"
    ];

    for (const query of testQueries) {
        const analysis = analyzeQuery(query);
        console.log(`${chalk.bold("Query:")} "${query}"`);
        console.log(`${chalk.dim("Type:")} ${analysis.queryType}`);
        console.log(`${chalk.dim("Weights:")} Vector ${analysis.weights.vector}/Text ${analysis.weights.text}\n`);

        const queryEmbedding = await embeddingContext.getEmbeddingFor(query);
        const results = await vectorStore.hybridSearch(
            NS,
            Array.from(queryEmbedding.vector),
            query,
            {
                vectorWeight: analysis.weights.vector,
                textWeight: analysis.weights.text,
                k: 2
            }
        );

        results.forEach((doc, idx) => {
            console.log(`  ${idx + 1}. [${chalk.cyan(doc.combinedScore.toFixed(4))}] ${doc.metadata.title}`);
        });
        console.log();
    }

    console.log(`${chalk.bold("💡 Key Insight:")}`);
    console.log("Query patterns guide weight selection:");
    console.log("• Product codes → Keyword-heavy (0.2/0.8)");
    console.log("• Technical specs → Keyword-heavy (0.3/0.7)");
    console.log("• Questions → Vector-heavy (0.8/0.2)");
    console.log("• Short keywords → Keyword-leaning (0.4/0.6)");
    console.log("• Mixed/default → Balanced (0.5/0.5)");
    console.log("→ Automated strategy improves UX!\n");
}

/**
 * Example 5: Handling Zero-Result Scenarios
 * Demonstrates fallback strategies when searches fail
 */
async function example5(embeddingContext) {
    const vectorStore = new VectorDB({ dim: DIM, maxElements: MAX_ELEMENTS });
    const products = createProductCatalog();

    await addProductsToStore(vectorStore, embeddingContext, products);

    const query = "Microsoft Surface laptop touchscreen";
    console.log(`${chalk.bold("Query:")} "${query}"`);
    console.log(chalk.dim("No Microsoft products in catalog!\n"));

    const queryEmbedding = await embeddingContext.getEmbeddingFor(query);

    // Strategy 1: Pure keyword (likely fails)
    console.log(chalk.bold("Strategy 1: Pure Keyword Search"));
    const keywordOnly = await vectorStore.fullTextSearch(NS, query, 3);
    console.log(`Results: ${keywordOnly.length > 0 ? keywordOnly.length : chalk.red("0 (failed)")}`);
    if (keywordOnly.length > 0) {
        keywordOnly.forEach((doc, idx) => {
            console.log(`  ${idx + 1}. ${doc.metadata.title}`);
        });
    }

    // Strategy 2: Hybrid with reasonable threshold
    console.log(`\n${chalk.bold("Strategy 2: Hybrid Search (balanced)")}`);
    const hybrid = await vectorStore.hybridSearch(
        NS,
        Array.from(queryEmbedding.vector),
        query,
        { k: 3 }
    );
    console.log(`Results: ${hybrid.length}`);
    hybrid.forEach((doc, idx) => {
        console.log(`  ${idx + 1}. [${chalk.cyan(doc.combinedScore.toFixed(4))}] ${doc.metadata.title}`);
        console.log(`     ${chalk.dim("Fallback: Similar to 'laptop touchscreen'")}`);
    });

    // Strategy 3: Vector-heavy fallback
    console.log(`\n${chalk.bold("Strategy 3: Vector-Heavy Fallback (0.9/0.1)")}`);
    const vectorFallback = await vectorStore.hybridSearch(
        NS,
        Array.from(queryEmbedding.vector),
        query,
        {
            vectorWeight: 0.9,
            textWeight: 0.1,
            k: 3
        }
    );
    console.log(`Results: ${vectorFallback.length}`);
    vectorFallback.forEach((doc, idx) => {
        console.log(`  ${idx + 1}. [${chalk.cyan(doc.combinedScore.toFixed(4))}] ${doc.metadata.title}`);
    });

    console.log(`\n${chalk.bold("💡 Fallback Strategy:")}`);
    console.log("When keyword search fails:");
    console.log("1. Start with balanced hybrid (0.5/0.5)");
    console.log("2. If few results, shift to vector-heavy (0.8/0.2)");
    console.log("3. For zero results, use pure vector (1.0/0.0)");
    console.log("4. Show 'similar products' messaging to user");
    console.log("5. Suggest query refinement or show popular items\n");
}

/**
 * Example 6: Price and Filter-Aware Hybrid Search
 * Combines hybrid search with business logic
 */
async function example6(embeddingContext) {
    const vectorStore = new VectorDB({ dim: DIM, maxElements: MAX_ELEMENTS });
    const products = createProductCatalog();

    await addProductsToStore(vectorStore, embeddingContext, products);

    const query = "high-end laptop professional work";
    const maxPrice = 2500;

    console.log(`${chalk.bold("Query:")} "${query}"`);
    console.log(`${chalk.bold("Max Price:")} $${maxPrice}\n`);

    const queryEmbedding = await embeddingContext.getEmbeddingFor(query);

    // Without price filter
    console.log(chalk.bold("Without Price Filter:"));
    const unfiltered = await vectorStore.hybridSearch(
        NS,
        Array.from(queryEmbedding.vector),
        query,
        { k: 5 }
    );

    unfiltered.forEach((doc, idx) => {
        const overBudget = doc.metadata.price > maxPrice;
        console.log(`  ${idx + 1}. [${chalk.cyan(doc.combinedScore.toFixed(4))}] ${doc.metadata.title}`);
        console.log(`     $${doc.metadata.price} ${overBudget ? chalk.red("(over budget)") : chalk.green("✓")}`);
    });

    // With price filter
    console.log(`\n${chalk.bold("With Price Filter (<= $2500):")}`);
    const filtered = await vectorStore.hybridSearch(
        NS,
        Array.from(queryEmbedding.vector),
        query,
        { k: 5 }
    );

    // Apply post-filtering (in production, do pre-filtering for efficiency)
    const priceFiltered = filtered.filter(doc => doc.metadata.price <= maxPrice);

    priceFiltered.forEach((doc, idx) => {
        console.log(`  ${idx + 1}. [${chalk.cyan(doc.combinedScore.toFixed(4))}] ${doc.metadata.title}`);
        console.log(`     $${doc.metadata.price} ${chalk.green("✓")}`);
    });

    // Category + Price filter
    console.log(`\n${chalk.bold("Category + Price Filter (laptops <= $2500):")}`);
    const categoryFiltered = filtered.filter(doc =>
        doc.metadata.category === 'laptops' && doc.metadata.price <= maxPrice
    );

    categoryFiltered.forEach((doc, idx) => {
        console.log(`  ${idx + 1}. [${chalk.cyan(doc.combinedScore.toFixed(4))}] ${doc.metadata.title}`);
        console.log(`     $${doc.metadata.price} | ${doc.metadata.category}`);
    });

    console.log(`\n${chalk.bold("💡 Key Insight:")}`);
    console.log("Combine hybrid search with business filters:");
    console.log("• Price ranges (budgets)");
    console.log("• Category constraints");
    console.log("• In-stock status");
    console.log("• User permissions/visibility");
    console.log("→ Search score + business rules = better UX\n");
}

/**
 * Example 7: Performance Optimization Strategies
 * Demonstrates caching and efficiency techniques
 */
async function example7(embeddingContext) {
    const vectorStore = new VectorDB({ dim: DIM, maxElements: MAX_ELEMENTS });
    const products = createProductCatalog();

    await addProductsToStore(vectorStore, embeddingContext, products);

    // Simulate query cache
    const queryCache = new Map();

    async function cachedHybridSearch(query, options = {}) {
        const cacheKey = `${query}:${JSON.stringify(options)}`;

        if (queryCache.has(cacheKey)) {
            return { results: queryCache.get(cacheKey), cached: true };
        }

        const queryEmbedding = await embeddingContext.getEmbeddingFor(query);
        const results = await vectorStore.hybridSearch(
            NS,
            Array.from(queryEmbedding.vector),
            query,
            options
        );

        queryCache.set(cacheKey, results);
        return { results, cached: false };
    }

    console.log(chalk.bold("Testing Query Cache:"));

    const testQuery = "wireless headphones noise cancelling";

    // First call
    console.log(`\n${chalk.dim("First search (cache miss)...")}`);
    const start1 = Date.now();
    const { results: results1, cached: cached1 } = await cachedHybridSearch(testQuery, { k: 3 });
    const time1 = Date.now() - start1;

    console.log(`Cached: ${chalk.red("No")}`);
    console.log(`Time: ${time1}ms`);
    results1.forEach((doc, idx) => {
        console.log(`  ${idx + 1}. ${doc.metadata.title}`);
    });

    // Second call (same query)
    console.log(`\n${chalk.dim("Second search (cache hit)...")}`);
    const start2 = Date.now();
    const { results: results2, cached: cached2 } = await cachedHybridSearch(testQuery, { k: 3 });
    const time2 = Date.now() - start2;

    console.log(`Cached: ${chalk.green("Yes")}`);
    console.log(`Time: ${time2}ms`);
    console.log(`Speedup: ${chalk.green(Math.round(time1/time2) + "x")}`);

    console.log(`\n${chalk.bold("Other Optimization Strategies:")}\n`);

    console.log(chalk.bold("1. Two-Stage Retrieval:"));
    console.log("   • Stage 1: Fast keyword search (100 candidates)");
    console.log("   • Stage 2: Vector rerank top candidates");
    console.log("   • Reduces expensive embedding calculations\n");

    console.log(chalk.bold("2. Index Partitioning:"));
    console.log("   • Separate indices by category");
    console.log("   • Search only relevant partitions");
    console.log("   • Example: electronics, clothing, books\n");

    console.log(chalk.bold("3. Approximate Search:"));
    console.log("   • Use HNSW for vector search");
    console.log("   • Trade accuracy for speed");
    console.log("   • Good for large catalogs (1M+ products)\n");

    console.log(chalk.bold("4. Query Result Caching:"));
    console.log("   • Cache popular queries");
    console.log("   • Set TTL based on update frequency");
    console.log("   • Invalidate on product updates\n");

    console.log(`${chalk.bold("💡 Production Checklist:")}`);
    console.log("✓ Implement query result caching");
    console.log("✓ Use two-stage retrieval for large catalogs");
    console.log("✓ Partition indices by category/domain");
    console.log("✓ Monitor slow queries and optimize");
    console.log("✓ Pre-compute embeddings for products");
    console.log("✓ Consider approximate nearest neighbor search\n");
}

// ============================================================================
// MAIN RUNNER
// ============================================================================

async function runAllExamples() {
    console.clear();
    console.log("\n" + "=".repeat(80));
    console.log(chalk.bold("RAG from Scratch - Hybrid Search (E-Commerce Focus)"));
    console.log("=".repeat(80) + "\n");

    console.log(chalk.dim("What you'll learn:"));
    console.log(chalk.dim("• Product search with SKUs and brand names"));
    console.log(chalk.dim("• Score normalization techniques"));
    console.log(chalk.dim("• Multi-field search strategies"));
    console.log(chalk.dim("• Dynamic weight adjustment"));
    console.log(chalk.dim("• Fallback strategies and performance optimization\n"));

    try {
        const embeddingContext = await OutputHelper.withSpinner(
            "Loading embedding model...",
            () => initializeEmbeddingModel()
        );

        await OutputHelper.runExample("Example 1: SKU and Brand Challenges", () => example1(embeddingContext));
        await OutputHelper.runExample("Example 2: Score Normalization", () => example2(embeddingContext));
        await OutputHelper.runExample("Example 3: Multi-Field Search", () => example3(embeddingContext));
        await OutputHelper.runExample("Example 4: Dynamic Weights", () => example4(embeddingContext));
        await OutputHelper.runExample("Example 5: Fallback Strategies", () => example5(embeddingContext));
        await OutputHelper.runExample("Example 6: Filter-Aware Search", () => example6(embeddingContext));
        await OutputHelper.runExample("Example 7: Performance Optimization", () => example7(embeddingContext));

        console.log(chalk.bold.green("\n✅ All examples completed successfully!\n"));

        console.log(chalk.bold("Key Takeaways:"));
        console.log("• Hybrid search is critical for e-commerce product search");
        console.log("• SKUs and brand names require keyword matching");
        console.log("• Score normalization ensures fair combination");
        console.log("• Multi-field indexing captures all product dimensions");
        console.log("• Query analysis enables dynamic weight adjustment");
        console.log("• Fallback strategies handle zero-result scenarios");
        console.log("• Performance optimization is essential at scale\n");

        console.log(chalk.bold("E-Commerce Search Best Practices:"));
        console.log("┌────────────────────────────┬─────────────────┐");
        console.log("│ Query Type                 │ Strategy        │");
        console.log("├────────────────────────────┼─────────────────┤");
        console.log("│ Product code (SKU)         │ Keyword (0.2/0.8)│");
        console.log("│ Brand + category           │ Keyword (0.3/0.7)│");
        console.log("│ Natural language question  │ Vector (0.8/0.2) │");
        console.log("│ Descriptive search         │ Balanced (0.5/0.5)│");
        console.log("│ Technical specifications   │ Keyword (0.3/0.7)│");
        console.log("└────────────────────────────┴─────────────────┘\n");

        console.log(chalk.bold("Production Recommendations:"));
        console.log("✓ Use multi-field indexing (title, brand, SKU, attributes)");
        console.log("✓ Implement query pattern detection for auto-weighting");
        console.log("✓ Add fallback to vector search for zero results");
        console.log("✓ Combine search with business filters (price, stock)");
        console.log("✓ Cache popular queries with appropriate TTL");
        console.log("✓ Monitor and optimize slow query patterns");
        console.log("✓ Consider two-stage retrieval for large catalogs\n");

    } catch (error) {
        console.error(chalk.red("\n❌ Error:"), error?.message ?? error);
        console.error(chalk.dim("\nMake sure you have:"));
        console.error(chalk.dim("1. Installed: npm install embedded-vector-db node-llama-cpp chalk"));
        console.error(chalk.dim("2. Model file: bge-small-en-v1.5.Q8_0.gguf in models/"));
        console.error(chalk.dim("3. Correct file structure and imports\n"));
    }

    process.exit(0);
}

runAllExamples();
import chalk from 'chalk';
import ora from 'ora';

/**
 * Handles output formatting, logging, and progress spinners
 * for text splitting examples.
 */
export class OutputHelper {
    static divider = '─'.repeat(80);

    // === SECTION: Console Formatting ===
    static createHeader(title, subtitle = '') {
        console.log(`\n${this.divider}\n${chalk.bold(title)}`);
        if (subtitle) console.log(chalk.dim(subtitle));
        console.log(`${this.divider}\n`);
    }

    static formatStats(stats) {
        console.log('\nStatistics:');
        for (const [key, val] of Object.entries(stats)) {
            console.log(`  ${key}: ${chalk.cyan(val)}`);
        }
        console.log();
    }

    static formatChunkPreview(chunk, index) {
        const meta = chunk.metadata;
        const info = [
            `Chunk ${meta.chunk + 1}/${meta.totalChunks}`,
            `Page ${meta.loc?.pageNumber ?? 'N/A'}`,
            `${chunk.pageContent.length} chars`
        ].join(' | ');
        const preview = chunk.pageContent.slice(0, 200).replace(/\n/g, ' ');
        console.log(`\n[Chunk ${index + 1}]\n${chalk.dim(info)}\n${preview}...\n`);
    }

    static formatStrategyComparison(name, chunks, avgSize, useCase) {
        console.log(`\n${chalk.bold(name)}\n  Result: ${chunks} chunks\n  Avg Size: ${avgSize} chars\n  Use Case: ${chalk.dim(useCase)}`);
    }

    // === SECTION: Logging ===
    static log = {
        success: msg => console.log(chalk.green('✓ ') + msg),
        info: msg => console.log(chalk.blue('ℹ ') + msg),
        warn: msg => console.log(chalk.yellow('⚠ ') + msg),
        error: msg => console.log(chalk.red('✗ ') + msg)
    };

    // === SECTION: Async Utilities ===
    static async withSpinner(message, fn) {
        const spinner = ora(message).start();
        try {
            const result = await fn();
            spinner.succeed();
            return result;
        } catch (err) {
            spinner.fail(err.message);
            throw err;
        }
    }

    // === SECTION: Stats ===
    static analyzeChunks(chunks) {
        const sizes = chunks.map(c => c.pageContent.length);
        const avg = Math.round(sizes.reduce((a, b) => a + b, 0) / sizes.length);
        return {
            avg,
            min: Math.min(...sizes),
            max: Math.max(...sizes),
            median: sizes.sort((a, b) => a - b)[Math.floor(sizes.length / 2)]
        };
    }

    // === SECTION: Example Runner ===
    static async runExample(title, fn, subtitle = '') {
        this.createHeader(title, subtitle);
        const t0 = Date.now();
        try {
            await fn();
            console.log(chalk.green(`\n✓ Completed in ${(Date.now() - t0) / 1000}s\n`));
        } catch (e) {
            console.error(chalk.red(`✗ Failed: ${e.message}`));
        }
    }
}

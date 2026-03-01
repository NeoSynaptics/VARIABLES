import {PDFParse} from 'pdf-parse';

/**
 * Base class for text splitting logic.
 * Inspired by LangChain.js, but simplified and optimized for readability.
 */
class Document {
    constructor(pageContent, metadata, id) {
        this.pageContent = pageContent;
        this.metadata = metadata;
        this.id = id;
    }
}

const cleanText = (text) => {
    return text
        // Remove page markers like "-- 1 of 22 --" or "— 5 of 12 —"
        .replace(/[-–—]\s*\d+\s*of\s*\d+\s*[-–—]/gi, '')
        // Normalize multiple spaces and tabs
        .replace(/[ \t]+/g, ' ')
        // Preserve paragraph breaks but remove unnecessary empty lines
        .replace(/\n{3,}/g, '\n\n')
        // Trim start/end spaces
        .trim();
}

const constructDocument = (pageContent, source, info, metadata, totalPages, pageNumber) => {
    return new Document(
        pageContent,
        {
            source,
            pdf: {
                info,
                metadata,
                totalPages,
            },
            loc: {
                pageNumber
            }
        },
        undefined
    );
}

const extractTextFromPDF = async (url, {splitPages} = {splitPages: false}) => {
    const parser = new PDFParse({url});
    const info = await parser.getInfo();
    const pages = info.total
    let doc = []

    if (splitPages) {
        for (let i = 0; i < pages; i++) {
            const rawText = (await parser.getText({partial: [i + 1]})).text;
            const cleanedText = cleanText(rawText);

            doc.push(constructDocument(
                cleanedText,
                url,
                info,
                info.metadata,
                pages,
                i + 1
            ))
        }
    } else {
        const rawText = (await parser.getText()).text;
        const cleanedText = cleanText(rawText);

        doc.push(constructDocument(
            cleanedText,
            url,
            info,
            info.metadata,
            pages,
        ))
    }

    return doc;
}

/**
 Retrieval-Augmented Generation for
 AI-Generated Content: A Survey
 https://arxiv.org/pdf/2402.19473

 A Comprehensive Survey of Retrieval-Augmented Generation (RAG): Evolution, Current
 Landscape and Future Directions
 https://arxiv.org/pdf/2410.12837

 Corrective Retrieval Augmented Generation
 https://arxiv.org/pdf/2401.15884
 */
const url = "https://arxiv.org/pdf/2402.19473"
extractTextFromPDF(url, {splitPages: true}).then(text => {
    console.log(`Extracted ${text.length} document(s). First page snippet:\n`);
    console.log(text[0]?.pageContent.slice(0, 300) + "...");
});


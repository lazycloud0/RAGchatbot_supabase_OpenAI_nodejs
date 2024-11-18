import fs from "fs";
import path from "path";
import { createClient } from "@supabase/supabase-js";
import { OpenAI } from "openai";
import { Document } from "langchain/schema";
import { CharacterTextSplitter } from "langchain/text_splitter";
import { SupabaseVectorStore } from "langchain/vectorstores";
import {
  UnstructuredMarkdownLoader,
  UnstructuredPDFLoader,
} from "langchain/document_loaders";
import dotenv from "dotenv";

// Load environment variables
dotenv.config();

// Initialize Supabase client
const url = process.env.SUPABASE_URL;
const key = process.env.SUPABASE_KEY;
const supabase = createClient(url, key);

// Initialize OpenAI client
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Function to load all .mdx files
async function loadAllMdxFiles(folderPath) {
  const mdxFiles = fs
    .readdirSync(folderPath)
    .filter((file) => file.endsWith(".mdx"));
  const documents = [];
  for (const file of mdxFiles) {
    const loader = new UnstructuredMarkdownLoader(path.join(folderPath, file));
    const loadedDocs = await loader.load();
    loadedDocs.forEach((doc) => {
      documents.push(
        new Document({ pageContent: doc.pageContent, metadata: doc.metadata })
      );
    });
  }
  return documents;
}

// Function to load all .pdf files
async function loadAllPdfFiles(folderPath) {
  const pdfFiles = fs
    .readdirSync(folderPath)
    .filter((file) => file.endsWith(".pdf"));
  const documents = [];
  for (const file of pdfFiles) {
    const loader = new UnstructuredPDFLoader(path.join(folderPath, file));
    const loadedDocs = await loader.load();
    loadedDocs.forEach((doc) => {
      documents.push(
        new Document({ pageContent: doc.pageContent, metadata: doc.metadata })
      );
    });
  }
  return documents;
}

// Function to load all transcripts
async function loadAllTranscripts(folderPath) {
  const transcriptFiles = fs
    .readdirSync(folderPath)
    .filter((file) => file.endsWith(".txt"));
  const documents = [];
  for (const file of transcriptFiles) {
    const content = fs.readFileSync(path.join(folderPath, file), "utf-8");
    documents.push(
      new Document({ pageContent: content, metadata: { source: file } })
    );
  }
  return documents;
}

// Main function to load documents and process user query
async function main() {
  const mdxDocs = await loadAllMdxFiles("docs/mdx");
  const pdfDocs = await loadAllPdfFiles("docs/pdfs");
  const transcriptDocs = await loadAllTranscripts("docs/transcripts");

  // Combine all documents
  const allDocs = [...mdxDocs, ...pdfDocs, ...transcriptDocs];

  // Concatenate the content of all documents into a single string
  const allText = allDocs.map((doc) => doc.pageContent).join("\n\n");

  // Split the text into chunks
  const textSplitter = new CharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const splitTexts = textSplitter.splitText(allText);

  // Create Document objects from split texts
  const docs = splitTexts.map((text) => new Document({ pageContent: text }));

  // Initialize the Supabase Vector Store
  const vectorStore = await SupabaseVectorStore.fromDocuments(docs, supabase, {
    tableName: "documents",
    queryName: "match_documents",
  });

  // AI chat
  const userQuery = prompt("Ask a question: "); // Use a suitable method to get user input

  const matchedDocs = await vectorStore.similaritySearch(userQuery);
  const injectedDocs = matchedDocs.map((doc) => doc.pageContent).join("\n\n");

  const completionMessages = [
    {
      role: "system",
      content:
        "You are an AI assistant with unparalleled expertise in Frontend engineering and javascript, possessing a profound understanding of the intricacies of React. Your primary task is to provide answers about web development, software development and frontend engineering topics using the documents provided below. You can comprehend and interpret information from any language. If a query is not addressed by these documents, you will utilize your extensive knowledge in javascript to provide accurate answers. Additionally, if users provide syntax and technical terms, you are equipped to suggest related topics and more information based on your understanding of javascript. Keep your responses concise and focused on the topics of javascript. And reference the relevant materials from the source.",
    },
    {
      role: "user",
      content: userQuery,
    },
    {
      role: "assistant",
      content: injectedDocs,
    },
  ];

  const response = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    messages: completionMessages,
    max_tokens: 150,
    temperature: 0.4,
  });

  console.log("Assistant: ", response.choices[0].message.content);
}

// Run the main function
main().catch(console.error);

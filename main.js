import fs from "fs";
import path from "path";
import { createClient } from "@supabase/supabase-js";
import { OpenAI } from "openai";
import { Document } from "@langchain/core/documents";
import { CharacterTextSplitter } from "langchain/text_splitter";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
// import {
//   UnstructuredMarkdownLoader,
//   PDFLoader,
// } from "langchain/community/document_loaders";
import readline from "readline";
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
// async function loadAllMdxFiles(folderPath) {
//   const mdxFiles = fs
//     .readdirSync(folderPath)
//     .filter((file) => file.endsWith(".mdx"));
//   const documents = [];
//   for (const file of mdxFiles) {
//     const loader = new UnstructuredMarkdownLoader(path.join(folderPath, file));
//     const loadedDocs = await loader.load();
//     loadedDocs.forEach((doc) => {
//       documents.push(
//         new Document({ pageContent: doc.pageContent, metadata: doc.metadata })
//       );
//     });
//   }
//   return documents;
// }

// Function to load all .pdf files
async function loadAllPdfFiles(folderPath) {
  const pdfFiles = fs
    .readdirSync(folderPath)
    .filter((file) => file.endsWith(".pdf"));
  const documents = [];
  for (const file of pdfFiles) {
    const loader = new PDFLoader(path.join(folderPath, file));
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

// Call the AI chat function within the main
// Main function to load documents and process user query
async function main() {
  //   const mdxDocs = await loadAllMdxFiles("docs/mdx");
  const pdfDocs = await loadAllPdfFiles("docs/pdfs");
  const transcriptDocs = await loadAllTranscripts("docs/transcripts");

  // Combine all documents
  //const allDocs = [...mdxDocs, ...pdfDocs, ...transcriptDocs];
  const allDocs = [...pdfDocs, ...transcriptDocs];

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

  // Start the AI chat
  await aiChat();
}

// Create an interface for reading input
const rl = readline.createInterface({
  input: process.stdin, // This is where we read input from
  output: process.stdout, // This is where we write output to
});

// Function to get user input
function askQuestion() {
  return new Promise((resolve) => {
    rl.question("Ask a question: ", (answer) => {
      resolve(answer); // Resolve the promise with the user's answer
      rl.close(); // Close the interface after getting the input
    });
  });
}

// AI chat
async function aiChat() {
  while (true) {
    const userQuery = await askQuestion(); // Use the new function to get user input

    if (userQuery.toLowerCase() === "exit") {
      console.log("Exiting the chat. Goodbye!");
      break; // Exit the loop if the user types 'exit'
    }

    // Perform a similarity search in the vector store using the user's query
    const matchedDocs = await vectorStore.similaritySearch(userQuery);

    // Combine the content of the matched documents into a single string
    const injectedDocs = matchedDocs.map((doc) => doc.pageContent).join("\n\n");

    // Prepare the messages for the OpenAI API
    const completionMessages = [
      {
        role: "system",
        content:
          "You are an AI assistant with unparalleled expertise in javascript, possessing a profound understanding of the intricacies of React. Your primary task is to provide answers about web development, software development topics using the documents provided. You can comprehend and interpret information from any language. If a query is not addressed by these documents, you will utilize your extensive knowledge in javascript to provide accurate answers. Additionally, if users provide syntax and technical terms, you are equipped to suggest related topics and more information based on your understanding of javascript. Keep your responses concise and focused on the topics of javascript.",
      },
      {
        role: "user",
        content: userQuery, // The user's question
      },
      {
        role: "assistant",
        content: injectedDocs, // The content from the matched documents
      },
    ];

    // Call the OpenAI API to get a response
    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo", // Specify the model to use
      messages: completionMessages, // Pass the prepared messages
      max_tokens: 150, // Limit the number of tokens in the response
      temperature: 0.4, // Control the randomness of the output
    });

    // Output the assistant's response to the console
    console.log("Assistant: ", response.choices[0].message.content);
  }
}
// Run the main function
main().catch(console.error);

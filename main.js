import fs from "fs";
import path from "path";
import { createClient } from "@supabase/supabase-js";
import { OpenAI } from "openai";
import { Document } from "@langchain/core/documents";
//import { CharacterTextSplitter } from "langchain/text_splitter";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { UnstructuredLoader } from "@langchain/community/document_loaders/fs/unstructured";
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

// Initialize Supabase Vector Store
//const embeddings = new OpenAIEmbeddings({ apiKey: process.env.OPENAI_API_KEY });

// Initialize Unstructured API key
//const unstructured_api_key = process.env.UNSTRUCTURED_API_KEY;

// Function to load all .mdx files
async function loadAllMdxFiles(folderPath) {
  console.log(`Reading files from folder: ${folderPath}`);
  const mdxFiles = fs
    .readdirSync(folderPath)
    .filter((file) => file.endsWith(".mdx"));
  console.log(`Found ${mdxFiles.length} MDX files.`);

  const documents = new Set();
  if (mdxFiles.length === 0) return [];

  for (const file of mdxFiles) {
    console.log(`Processing file: ${file}`);
    const loader = new UnstructuredMarkdownLoader(path.join(folderPath, file), {
      apiKey: process.env.UNSTRUCTURED_API_KEY, // Provide the API key here
    });

    try {
      const loadedDocs = await loader.load();
      console.log(`Loaded ${loadedDocs.length} documents from file: ${file}`);

      loadedDocs.forEach((doc) => {
        const docKey = JSON.stringify({
          pageContent: doc.pageContent,
          metadata: doc.metadata,
        });
        documents.add(docKey);
      });
    } catch (error) {
      console.error(`Error loading file ${file}:`, error);
    }
  }

  return Array.from(documents).map((docKey) => JSON.parse(docKey));
}

// Function to load all .pdf files
async function loadAllPdfFiles(folderPath) {
  const pdfFiles = fs
    .readdirSync(folderPath)
    .filter((file) => file.endsWith(".pdf"));
  const documents = new Set();
  if (pdfFiles.length === 0) return [];
  for (const file of pdfFiles) {
    const loader = new PDFLoader(path.join(folderPath, file));
    const loadedDocs = await loader.load();
    loadedDocs.forEach((doc) => {
      const docKey = JSON.stringify({
        pageContent: doc.pageContent,
        metadata: doc.metadata,
      });
      documents.add(docKey);
    });
  }
  return Array.from(documents).map((docKey) => JSON.parse(docKey));
}

// Function to load all transcripts
async function loadAllTranscripts(folderPath) {
  const transcriptFiles = fs
    .readdirSync(folderPath)
    .filter((file) => file.endsWith(".txt"));
  const documents = new Set();
  if (transcriptFiles.length === 0) return [];
  for (const file of transcriptFiles) {
    const content = fs.readFileSync(path.join(folderPath, file), "utf-8");
    const docKey = JSON.stringify({
      pageContent: content,
      metadata: { fileName: file },
    });
    documents.add(docKey);
  }
  return Array.from(documents).map((docKey) => JSON.parse(docKey));
}

// Function for vector embeddings
async function createEmbeddings(docs) {
  try {
    const response = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: docs,
    });

    // Log the entire response for inspection
    console.log("OpenAI API response:", JSON.stringify(response, null, 2));

    // Check if the response contains the expected data
    if (!response || !response.data || !Array.isArray(response.data)) {
      throw new Error("Invalid response from OpenAI API");
    }

    // Extract embeddings
    return response.data.map((item) => item.embedding);
  } catch (err) {
    // Improved error handling
    if (err.response) {
      // If the error has a response, log the response details
      console.error(
        "OpenAI API error response:",
        JSON.stringify(err.response.data, null, 2)
      );
      console.error(
        "Error message:",
        err.response.data.error
          ? err.response.data.error.message
          : "No error message available"
      );
    } else {
      // Log the error if it doesn't have a response
      console.error("Error creating embeddings:", err.message);
    }
    throw err; // Rethrow the error after logging
  }
}

// Call the AI chat function within the main
// Main function to load documents and process user query
async function main() {
  console.log("Starting main function...");
  // Load documents
  // console.log("Loading MDX documents...");
  // const mdxDocs = await loadAllMdxFiles("docs/mdx");
  // console.log(`Loaded ${mdxDocs.length} MDX documents.`);

  console.log("Loading PDF documents...");
  const pdfDocs = await loadAllPdfFiles("docs/pdfs");
  console.log(`Loaded ${pdfDocs.length} PDF documents.`);

  console.log("Loading transcript documents...");
  const transcriptDocs = await loadAllTranscripts("docs/transcripts");
  console.log(`Loaded ${transcriptDocs.length} transcript documents.`);

  // Combine all documents
  //const allDocs = [...mdxDocs, ...pdfDocs, ...transcriptDocs];
  const allDocs = [...pdfDocs, ...transcriptDocs];
  console.log(`Total documents to process: ${allDocs.length}`);

  // Check if there are any documents to process
  if (allDocs.length === 0) {
    console.log("No documents found to process.");
    return;
  }

  // Concatenate the content of all documents into a single string
  console.log("Concatenating document contents...");
  const allText = allDocs.map((doc) => doc.pageContent).join("\n\n");

  // Custom function to split text into chunks of specified size
  function splitTextIntoChunks(text, chunkSize, chunkOverlap) {
    const chunks = [];
    for (let i = 0; i < text.length; i += chunkSize - chunkOverlap) {
      const chunk = text.slice(i, i + chunkSize);
      chunks.push(chunk);
    }
    return chunks;
  }

  // Split the text into chunks
  console.log("Splitting text into chunks...");
  const chunkSize = 1000;
  const chunkOverlap = 200;
  let splitTexts = splitTextIntoChunks(allText, chunkSize, chunkOverlap);
  console.log(`Initial split resulted in ${splitTexts.length} chunks.`);

  // Further split any chunks that exceed the chunk size
  console.log("Further splitting large chunks...");
  splitTexts = splitTexts.flatMap((text) => {
    if (text.length > chunkSize) {
      return splitTextIntoChunks(text, chunkSize, chunkOverlap);
    }
    return [text];
  });
  console.log(`Final split resulted in ${splitTexts.length} chunks.`);

  // Create Document objects from split texts
  // Create Document objects from split texts
  console.log("Creating Document objects...");
  const docs = splitTexts.map((text) => new Document({ pageContent: text }));
  console.log(`Created ${docs.length} Document objects.`);

  // Create embeddings for the documents
  console.log("Creating embeddings for the documents...");
  for (const doc of docs) {
    try {
      const embeddings = await createEmbeddings([doc.pageContent]);
      doc.embedding = embeddings[0]; // Assuming createEmbeddings returns an array
    } catch (err) {
      console.error(
        `Error creating embeddings for document: ${doc.pageContent}`,
        err
      );
    }
  }

  // Function to batch requests to check for existing embeddings
  //

  // // Save the documents to the vector store via SupabaseVectorStore
  // console.log("Saving documents to the vector store...");
  // const vectorStore = new SupabaseVectorStore({
  //   supabase,
  //   tableName: "vectorstore",
  // });
  // await vectorStore.addDocuments(docs);
  // console.log("Documents saved to the vector store.");

  // Initialize an array to store the results
  const saveResults = [];

  // Save the documents to the vector store
  for (const doc of docs) {
    const { data, error } = await supabase.from("documents").insert({
      content: doc.pageContent,
      embedding: doc.embedding,
    });

    // Save the result of each insertion
    saveResults.push({ data, error });

    if (error) {
      console.log("Error saving document to the vector store: ", error);
    }
  }
  console.log("Documents saved to the vector store.");

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
    });
  });
}

// AI chat
async function aiChat() {
  console.log("Starting AI chat. Type 'exit' to end the chat.");

  while (true) {
    const userQuery = await askQuestion();

    // Exit the chat if the user types 'exit'
    if (userQuery.toLowerCase() === "exit") {
      console.log("Ending chat. Goodbye!");
      rl.close();
      break;
    }

    // Create an embedding for the user's query
    let queryEmbedding;
    try {
      queryEmbedding = await createEmbeddings([userQuery]);
    } catch (err) {
      console.error("Error creating embedding for user query:", err);
      continue; // Skip to the next iteration if there's an error
    }

    // Perform a similarity search in the Supabase vector store using the match_documents function
    const { data: similarDocs, error } = await supabase.rpc("match_documents", {
      query_embedding: queryEmbedding[0],
    }); // Call the RPC function

    if (error) {
      console.error("Error during similarity search:", error);
      continue; // Skip to the next iteration if there's an error
    }

    // Check if any similar documents were found
    if (similarDocs.length === 0) {
      console.log("No similar documents found.");
      continue; // Skip to the next iteration if no documents are found
    }

    // Log the similar documents
    // console.log("Similar documents found:");
    // similarDocs.forEach((doc, index) => {
    //   console.log(
    //     `${index + 1}: ${doc.content} (Similarity: ${doc.similarity})`
    //   );
    // });

    // Prepare the messages for the OpenAI API
    const injectedDocs = similarDocs.map((doc) => doc.content).join("\n"); // Combine the content from matched documents
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

    // Generate a response from OpenAI using the prepared messages
    try {
      const aiResponse = await openai.chat.completions.create({
        model: "gpt-3.5-turbo", // or any other model you want to use
        messages: completionMessages,
        max_tokens: 150, // Limit the number of tokens in the response
        temperature: 0.4, // Control the randomness of the output
      });

      // Log the AI's response
      console.log("AI:", aiResponse.choices[0].message.content);
    } catch (err) {
      console.error("Error during AI response generation:", err);
    }
  }
}

main().catch(console.error);

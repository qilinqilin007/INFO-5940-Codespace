Step 1: Set up the API_KEY and run the streamlit app
Type the commands below in terminal:

API_KEY="your own API KEY" streamlit run assign1_5940.py
Step 2: Set up model, chunking strategy, and temperature in sidebar.

Step 3: Upload documents.

Step 4:Choose the Retrieve Documents button.

Step 5: Then type the question in the bottom box to ask queestion.



Overview of Application Features
RAG Chat Box 🤖

An intelligent document question-answering system based on Streamlit + LangChain + OpenAI, which supports uploading documents, building knowledge bases, and interactive question-answering with LLM.
Front-end Page & Functional Modules
1. Sidebar Settings

-ChatGPT Model: Use the drop-down menu to select a different model (e.g., gpt-4o, gpt-4o-mini).
-Chunking Strategy: Select a document chunking strategy (recursive, character, markdown).
-Temperature: Use the slider to adjust the randomness of the generated responses (default 0.2, for a more robust approach).
Highlights: Users can flexibly select models and parameters to balance accuracy and creativity.

2. Main Page

(1) Document Upload Area
- Supports drag-and-drop upload or browsing to select files.
- File formats: .txt / .pdf
- Single file limit: 200MB.
Highlights: Supports large file uploads, meeting learning/research/internal enterprise document scenarios.
(2) Operation Buttons
- Retrieve Documents Button
Build index: Split uploaded documents → Embed and vectorize → Store in database.
- Clear Conversation Button
Clears the conversation, uploaded documents, and index, restoring the original state.
Highlights: Simple and intuitive button operation, supporting "one-click search/one-click clear".
(3) Status Prompt Area
- After a successful upload, the number of documents and the number of blocks will be displayed.
- If the upload fails or the file is not processed, a reminder message will be displayed.
(4) Conversation Window
- The user enters a question at the bottom (natural language questions are supported).
- The system returns the answer and lists the corresponding document source in "Sources".
- Supports continuous conversation and maintains context memory.
Highlights: It not only accurately retrieves answers, but also automatically summarizes documents and always provides sources to ensure transparent results.

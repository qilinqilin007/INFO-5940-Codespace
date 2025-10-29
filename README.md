Step 1: Set up the API_KEY and run the streamlit app
Type the commands below in terminal:

API_KEY="your own API KEY" streamlit run assign1_5940.py
![alt text](6da6863f-40e6-4cbe-afd0-762995b8272a.png)
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
![alt text](b2f36775-9dd7-40a3-9bd7-3cd457cdf1d4.png)
-Chunking Strategy: Select a document chunking strategy (recursive, character, markdown).
![alt text](94034322-e9a8-41fc-8de0-062b71648ddf.png)
-Temperature: Use the slider to adjust the randomness of the generated responses (default 0.2, for a more robust approach).
![alt text](84b9f688-a8b2-4db9-b57e-7bf64eaea22d.png)
Highlights: Users can flexibly select models and parameters to balance accuracy and creativity.

2. Main Page

(1) Document Upload Area
- Supports drag-and-drop upload or browsing to select files.
- File formats: .txt / .pdf
- Single file limit: 200MB.
Highlights: Supports large file uploads, meeting learning/research/internal enterprise document scenarios.
![alt text](3d520375-2f4c-4f57-9900-d9d66bfcbf6f.png)
(2) Operation Buttons
- Retrieve Documents Button
Build index: Split uploaded documents → Embed and vectorize → Store in database.
- Clear Conversation Button
Clears the conversation, uploaded documents, and index, restoring the original state.
Highlights: Simple and intuitive button operation, supporting "one-click search/one-click clear".
![alt text](9041ca1e-318e-4352-8bc4-16f63e3ab0bd.png)
(3) Status Prompt Area
- After a successful upload, the number of documents and the number of blocks will be displayed.
- If the upload fails or the file is not processed, a reminder message will be displayed.
![alt text](8c41c09d-5c9c-44e5-9484-94175c61a007.png)
(4) Conversation Window
- The user enters a question at the bottom (natural language questions are supported).
![alt text](5ed8ba71-538b-4957-880c-9e82300cba39.png)
- The system returns the answer and lists the corresponding document source in "Sources".
![alt text](635f3a78-2a60-4157-984a-ddd56148376f.png)
- Supports continuous conversation and maintains context memory.
Highlights: It not only accurately retrieves answers, but also automatically summarizes documents and always provides sources to ensure transparent results.
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import uuid
from datetime import datetime

class ConversationStore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.conversation_store = Chroma(
            collection_name="conversation_history",
            embedding_function=self.embeddings,
            persist_directory="./conversation_db"
        )
    
    def add_conversation(self, question, answer, rag_type):
        # Create unique ID for the conversation
        conversation_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Combine question and answer for embedding
        conversation_text = f"Question: {question}\nAnswer: {answer}"
        
        # Store with metadata
        self.conversation_store.add_texts(
            texts=[conversation_text],
            metadatas=[{
                "conversation_id": conversation_id,
                "timestamp": timestamp,
                "rag_type": rag_type,
                "type": "conversation",
                "question": question,
                "answer": answer
            }],
            ids=[conversation_id]
        )
        self.conversation_store.persist()
    
    def get_relevant_history(self, query, rag_type, k=5):
        # Search for relevant conversations
        results = self.conversation_store.similarity_search_with_metadata(
            query=query,
            k=k,
            filter={"type": "conversation", "rag_type": rag_type}
        )
        
        # Format results
        history = []
        for doc in results:
            history.append({
                "question": doc.metadata["question"],
                "answer": doc.metadata["answer"],
                "timestamp": doc.metadata["timestamp"]
            })
        
        # Sort by timestamp
        history.sort(key=lambda x: x["timestamp"])
        return history 
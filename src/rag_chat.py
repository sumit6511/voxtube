"""
RAG (Retrieval-Augmented Generation) Chat Module
Enables Q&A over YouTube comments using vector search + LLM
"""

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import google.generativeai as genai
from config import EMBEDDING_MODEL, GEMINI_API_KEY, TOP_K_RETRIEVAL


class CommentVectorStore:
    """Vector store for comment embeddings using FAISS"""
    
    def __init__(self, embedding_model: str = None):
        self.embedding_model_name = embedding_model or EMBEDDING_MODEL
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.index = None
        self.comments = []
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
    
    def add_comments(self, comments: List[Dict], text_key: str = 'cleaned_text'):
        """
        Add comments to vector store
        
        Args:
            comments: List of comment dictionaries
            text_key: Key containing the text to embed
        """
        # Filter valid comments
        valid_comments = [c for c in comments if c.get(text_key, '').strip()]
        
        if not valid_comments:
            return
        
        # Generate embeddings
        texts = [c[text_key] for c in valid_comments]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index if not exists
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
        else:
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
        
        # Store comments
        self.comments.extend(valid_comments)
    
    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
        """
        Search for similar comments
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of similar comments with similarity scores
        """
        if self.index is None or len(self.comments) == 0:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.comments)))
        
        # Return results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.comments):
                comment = self.comments[idx].copy()
                comment['similarity_score'] = float(score)
                results.append(comment)
        
        return results
    
    def clear(self):
        """Clear the vector store"""
        self.index = None
        self.comments = []


class RAGChatbot:
    """
    RAG-based chatbot for Q&A over YouTube comments
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Vector store
        self.vector_store = CommentVectorStore()
        
        # Chat history
        self.history = []
    
    def ingest_comments(self, comments: List[Dict]):
        """Ingest comments into the vector store"""
        self.vector_store.add_comments(comments)
        print(f"Ingested {len(comments)} comments into vector store")
    
    def create_prompt(self, query: str, retrieved_comments: List[Dict]) -> str:
        """
        Create a prompt for the LLM with retrieved context
        
        Args:
            query: User question
            retrieved_comments: Retrieved relevant comments
        
        Returns:
            Formatted prompt
        """
        # Format retrieved comments
        context_parts = []
        for i, comment in enumerate(retrieved_comments[:5], 1):
            text = comment.get('text', '')
            author = comment.get('author', 'Anonymous')
            sentiment = comment.get('sentiment', 'neutral')
            likes = comment.get('like_count', 0)
            context_parts.append(
                f"[{i}] Author: {author} | Sentiment: {sentiment} | Likes: {likes}\n"
                f"Comment: {text[:200]}{'...' if len(text) > 200 else ''}"
            )
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are VoxTube AI, an intelligent assistant that helps content creators understand their YouTube audience by analyzing comments.

Based on the following relevant comments retrieved from the video, please answer the user's question.

RETRIEVED COMMENTS:
{context}

USER QUESTION: {query}

Instructions:
1. Answer based ONLY on the retrieved comments provided above
2. Be concise but informative
3. If the comments don't contain enough information, say so clearly
4. Cite comment numbers [1], [2], etc. when referencing specific comments
5. Provide insights about what the audience is saying

Your Answer:"""
        
        return prompt
    
    def chat(self, query: str, include_history: bool = False) -> Dict:
        """
        Process a chat query using RAG
        
        Args:
            query: User question
            include_history: Whether to include chat history
        
        Returns:
            Dictionary with answer and retrieved comments
        """
        # Retrieve relevant comments
        retrieved = self.vector_store.search(query, top_k=TOP_K_RETRIEVAL)
        
        if not retrieved:
            return {
                'answer': "I couldn't find any relevant comments to answer your question. Try asking something about the video content, quality, or audience feedback.",
                'retrieved_comments': [],
                'sources': []
            }
        
        # Create prompt
        prompt = self.create_prompt(query, retrieved)
        
        # Generate response
        try:
            response = self.model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            answer = f"Error generating response: {str(e)}"
        
        # Format sources
        sources = []
        for comment in retrieved[:3]:
            sources.append({
                'author': comment.get('author', 'Anonymous'),
                'text': comment.get('text', '')[:100] + '...' if len(comment.get('text', '')) > 100 else comment.get('text', ''),
                'sentiment': comment.get('sentiment', 'neutral'),
                'similarity': round(comment.get('similarity_score', 0), 3)
            })
        
        # Update history
        self.history.append({'role': 'user', 'content': query})
        self.history.append({'role': 'assistant', 'content': answer})
        
        return {
            'answer': answer,
            'retrieved_comments': retrieved,
            'sources': sources
        }
    
    def clear_history(self):
        """Clear chat history"""
        self.history = []
    
    def get_suggested_questions(self) -> List[str]:
        """Get suggested questions for the user"""
        return [
            "What do viewers like most about this video?",
            "What are the main complaints or issues mentioned?",
            "How is the audio/video quality according to viewers?",
            "What topics are viewers discussing the most?",
            "Are there any toxic or negative comments I should be aware of?",
            "What suggestions do viewers have for improvement?",
            "How does sentiment vary throughout the comments?"
        ]

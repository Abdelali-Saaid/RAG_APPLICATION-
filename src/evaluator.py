from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.core import Settings

class RAGEvaluator:
    """Handles Evaluation of RAG responses."""
    
    @staticmethod
    def evaluate(query: str, response_obj):
        """Evaluate the accuracy and faithfulness of a response."""
        llm = Settings.llm
        faith_eval = FaithfulnessEvaluator(llm=llm)
        rel_eval = RelevancyEvaluator(llm=llm)
        
        faith_result = faith_eval.evaluate_response(response=response_obj)
        rel_result = rel_eval.evaluate_response(query=query, response=response_obj)
        
        return {
            "faithfulness": faith_result.passing,
            "relevancy": rel_result.passing,
            "feedback": faith_result.feedback
        }

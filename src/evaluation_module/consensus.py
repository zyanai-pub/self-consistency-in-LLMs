from typing import List, Dict, Any

class ConsensusManager:
    """
    Implements majority voting and other consensus mechanisms for self-consistency.
    """
    def get_majority_vote(self, generated_paths: List[Dict[str, Any]]) -> str:
        """
        Takes a list of generated paths (dictionaries containing 'extracted_answer') 
        and returns the most frequent answer.
        """
        if not generated_paths:
            return ""
        
        vote_count = dict()
        for path in generated_paths:
            extracted = path.get('extracted_answer') 
            
            if extracted:
                if extracted not in vote_count:
                    vote_count[extracted] = 0
                vote_count[extracted] += 1

        # Handle no valid answers
        if not vote_count:
            return ""
        
        # Finiding tied answers
        max_votes = max(vote_count.values())
        max_voted_answers = [answer for answer in vote_count if vote_count.get(answer)==max_votes]

        # No tie
        if len(max_voted_answers) == 1:
            return max_voted_answers[0]
        
        # Tie breaking based on average confidence
        highest_avg_conf = float('-inf')
        best_answer = None
        
        for answer in max_voted_answers:
            answer_paths = [p for p in generated_paths if p.get('extracted_answer') == answer]
            
            confidence_scores = [p.get('confidence', 0.0) for p in answer_paths]
            avg_conf = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

            if avg_conf > highest_avg_conf:
                highest_avg_conf = avg_conf
                best_answer = answer

        return best_answer
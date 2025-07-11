import pandas as pd
import unicodedata
import re
from typing import List, Set, Dict, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gambling_detector")

@dataclass
class DetectionRules:
    """Configuration class for gambling detection rules and thresholds"""
    
    # Brand names commonly associated with gambling content
    known_brands: List[str] = None
    
    # Keywords that often appear in gambling promotional content
    promo_keywords: List[str] = None
    
    # Phrases commonly found in gambling testimonials
    testimonial_phrases: List[str] = None
    
    # Emojis often used in gambling-related content
    wealth_emojis: List[str] = None
    
    # Patterns that might indicate gambling content in usernames or comments
    spam_patterns: List[str] = None
    
    # Thresholds for severity classification
    high_severity_threshold: float = 0.7
    medium_severity_threshold: float = 0.4
    flagging_threshold: float = 0.5
    borderline_check_threshold: float = 0.4
    
    def __post_init__(self):
        """Initialize default values if none provided"""
        if self.known_brands is None:
            self.known_brands = [
                'dewa', 'dewadora', 'aero88', 'xl777', 'sgi88', 'alexis17',
                'slot', 'slotku', 'luxury88', 'parlay', 'jackpot', 'win88',
                'autokaya', 'gacor', 'weton88', 'dora77', 'aero', 'xl77',
                'pulau777', 'pulaυ777', 'gacor77', 'play777', 'miya88',
                # Additional patterns that may indicate gambling brands
                r'[a-z]{2,4}(77|88|99)', r'win\d{2,3}', r'kaya\d{2,3}', r'slot\d{2,3}'
            ]
            
        if self.promo_keywords is None:
            self.promo_keywords = [
                'jepe', 'jp', 'jackpot', 'untung', 'sultan', 'toto', 'slot',
                'auto liburan', 'modal sedikit', 'bonus new member', 'deposit kecil',
                'menang', 'hoki', 'juta', 'hadiah', 'cashback', 'bet', 'spin', 'scatter'
            ]
            
        if self.testimonial_phrases is None:
            self.testimonial_phrases = [
                'bisa beli', 'tidak menyangka', 'transformasi hidup', 'dari tukang',
                r'dari [\w\s]+ jadi', 'berubah hidup', 'bisa liburan', 'langsung kaya',
                'jadi sultan', 'hidup berubah', 'rezeki', 'coba dulu', 'cobain aja',
                'terbukti', 'teruji', 'testimoni', 'terpercaya', 'rekomendasi'
            ]
            
        if self.wealth_emojis is None:
            self.wealth_emojis = ['🤑', '💰', '🎰', '🏦', '✈', '🏡', '🍾', '🥳', '📈', '💸', '💎', '👑', '🏆', '💵', '💲', '🚗', '🏝️', '💯']
            
        if self.spam_patterns is None:
            self.spam_patterns = [
                r'[a-z]{2,10}[\s_]*(77|88|17|99|66)+',
                r'(d[oо0]r[aа4]|dew[aа4]|wet[oо0]n|pul[aа]u)[\s_]*(77|88|99)',
                r'[a-z]{3,}\d{2,}',
                r'(bonus|jp|jackpot|gacor|slot)[_\-. ]?[0-9]{1,3}',
                r'(mega|super|maxi)[_\-. ]?win',
                r'\d{1,3}[_\-. ]?[xX][_\-. ]?lipat'
            ]


class TextNormalizer:
    """Handles text normalization to handle obfuscation techniques"""
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalize Unicode characters to ASCII equivalents,
        handling stylized fonts and lookalike characters
        """
        if not isinstance(text, str):
            text = str(text)
            
        # Normalize accents and diacritics
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))

        # Manual fix for stylized letters and lookalikes
        translation_table = str.maketrans({
            '𝟎': '0', '𝟏': '1', '𝟐': '2', '𝟑': '3', '𝟒': '4',
            '𝟓': '5', '𝟔': '6', '𝟕': '7', '𝟖': '8', '𝟗': '9',
            '𝐴': 'A', '𝐵': 'B', '𝐶': 'C', '𝐷': 'D', '𝐸': 'E', '𝐹': 'F',
            '𝐺': 'G', '𝐻': 'H', '𝐼': 'I', '𝐽': 'J', '𝐾': 'K', '𝐿': 'L',
            '𝑀': 'M', '𝑁': 'N', '𝑂': 'O', '𝑃': 'P', '𝑄': 'Q', '𝑅': 'R',
            '𝑆': 'S', '𝑇': 'T', '𝑈': 'U', '𝑉': 'V', '𝑊': 'W', '𝑋': 'X',
            '𝑌': 'Y', '𝑍': 'Z',
            '𝘼': 'A', '𝘽': 'B', '𝘾': 'C', '𝘿': 'D', '𝙀': 'E', '𝙁': 'F',
            '𝙂': 'G', '𝙃': 'H', '𝙄': 'I', '𝙅': 'J', '𝙆': 'K', '𝙇': 'L',
            '𝙈': 'M', '𝙉': 'N', '𝙊': 'O', '𝙋': 'P', '𝙌': 'Q', '𝙍': 'R',
            '𝙎': 'S', '𝙏': 'T', '𝙐': 'U', '𝙑': 'V', '𝙒': 'W', '𝙓': 'X',
            '𝙔': 'Y', '𝙕': 'Z',
            'О': 'O', 'А': 'A', 'Я': 'R', 'а': 'a', 'о': 'o', 'р': 'p',  # Cyrillic
            '𝖆': 'a', '𝖇': 'b', '𝖈': 'c', '𝖉': 'd', '𝖊': 'e', '𝖋': 'f', '𝖌': 'g',  # extra Gothic styles
            '𝖍': 'h', '𝖎': 'i', '𝖏': 'j', '𝖐': 'k', '𝖑': 'l', '𝖒': 'm', '𝖓': 'n',
            '𝖔': 'o', '𝖕': 'p', '𝖖': 'q', '𝖗': 'r', '𝖘': 's', '𝖙': 't', '𝖚': 'u',
            '𝖛': 'v', '𝖜': 'w', '𝖝': 'x', '𝖞': 'y', '𝖟': 'z',
            # Additional lookalikes
            '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '7': 't',
        })

        # Translate and remove remaining non-ASCII
        text = text.translate(translation_table)
        return re.sub(r'[^\x00-\x7F]+', '', text)

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text for comparison:
        - Convert to lowercase
        - Remove punctuation and symbols
        - Normalize spacing
        """
        text = TextNormalizer.normalize_unicode(text)
        text = text.lower()
        
        # Save emojis for later analysis
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251" 
            "]+"
        )
        emojis = emoji_pattern.findall(text)
        
        # Remove punctuation but preserve emojis in a separate variable
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip(), emojis

    @staticmethod
    def collapse_brand_chars(text: str) -> str:
        """
        Collapse space-separated, dot-separated or stylized characters
        that might be used to obfuscate brand names
        """
        patterns = [
            # Handle space/dot/dash/underscore separators in letter sequences
            (r'(?:[a-zA-Z0-9][\s\.\-_]*){3,}', lambda m: m.group(0).replace(' ', '').replace('.', '').replace('-', '').replace('_', '')),
            # Handle special spacing patterns
            (r'([a-z])(\s+)([a-z])', r'\1\3'),
            # Handle zero-width spaces and other invisible separators
            (r'[a-z]\u200B[a-z]', lambda m: m.group(0).replace('\u200B', '')),
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
            
        return text


class GamblingDetector:
    """Main class for detecting gambling content in comments"""
    
    def __init__(self, rules: Optional[DetectionRules] = None):
        """Initialize with detection rules"""
        self.rules = rules or DetectionRules()
        self.text_normalizer = TextNormalizer()
        self.cached_usernames: Dict[str, float] = {}
        
    def contains_wealth_emoji(self, text: str, emojis: List[str]) -> bool:
        """Check if text contains wealth-related emojis"""
        return any(emoji in text for emoji in self.rules.wealth_emojis) or \
               any(emoji in emojis for emoji in self.rules.wealth_emojis)
               
    def contains_testimonial(self, text: str) -> bool:
        """Check if text contains testimonial phrases"""
        for phrase in self.rules.testimonial_phrases:
            if re.search(phrase, text):
                return True
        return False
        
    def contains_promo_keywords(self, text: str) -> bool:
        """Check if text contains promotional keywords"""
        words = set(text.split())
        return any(kw in text for kw in self.rules.promo_keywords) or \
               any(kw in words for kw in self.rules.promo_keywords)
               
    def check_brand_match(self, text: str) -> Tuple[bool, float]:
        """
        Check if text contains known gambling brand names,
        return a tuple of (matched, score)
        """
        # Direct match - higher confidence
        for brand in self.rules.known_brands:
            if isinstance(brand, str) and brand in text:
                return True, 0.6
            elif hasattr(brand, 'search') and brand.search(text):  # Regex pattern
                return True, 0.6
                
        # Check for brand without spaces - medium confidence
        no_spaces = text.replace(' ', '')
        for brand in self.rules.known_brands:
            if isinstance(brand, str) and brand in no_spaces:
                return True, 0.4
                
        return False, 0.0
        
    def check_spam_patterns(self, text: str) -> bool:
        """Check if text matches common spam patterns"""
        for pattern in self.rules.spam_patterns:
            if re.search(pattern, text):
                return True
        return False
        
    def score_username(self, username: str) -> float:
        """Score a username for likelihood of being a gambling promoter"""
        if username in self.cached_usernames:
            return self.cached_usernames[username]
            
        score = 0.0
        cleaned, _ = self.text_normalizer.clean_text(username)
        collapsed = self.text_normalizer.collapse_brand_chars(cleaned)
        
        # Check for brand names in username
        brand_match, brand_score = self.check_brand_match(collapsed)
        if brand_match:
            score += brand_score
            
        # Check for spam patterns in username
        if self.check_spam_patterns(collapsed):
            score += 0.3
            
        # Cache the result
        self.cached_usernames[username] = score
        return score
        
    def score_comment(self, comment: str, username: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate gambling score for a comment and associated username
        Returns a dictionary with score details
        """
        result = {
            'score': 0.0,
            'has_brand': False,
            'has_spam_pattern': False,
            'has_promo': False,
            'has_testimonial': False,
            'has_wealth_emoji': False
        }
        
        # Process and clean the comment text
        cleaned, emojis = self.text_normalizer.clean_text(comment)
        collapsed = self.text_normalizer.collapse_brand_chars(cleaned)
        
        # Check for brand names
        brand_match, brand_score = self.check_brand_match(collapsed)
        if brand_match:
            result['score'] += brand_score
            result['has_brand'] = True
            
        # Check for spam patterns
        if self.check_spam_patterns(collapsed):
            result['score'] += 0.3
            result['has_spam_pattern'] = True
            
        # Check for promotional language
        promo_words = ['jepey', 'main', 'bonus', 'gacor', 'slot', 'winrate', 'member', 'daftar']
        if any(word in collapsed for word in promo_words):
            result['score'] += 0.2
            result['has_promo'] = True
            
        # Check for testimonials
        if self.contains_testimonial(collapsed):
            result['score'] += 0.2
            result['has_testimonial'] = True
            
        # Check for wealth emojis in the original text
        if self.contains_wealth_emoji(comment, emojis):
            result['score'] += 0.25
            result['has_wealth_emoji'] = True
            
        # Username check if provided
        username_score = 0.0
        if username:
            username_score = self.score_username(username)
            result['username_score'] = username_score
            if username_score > 0:
                result['score'] += min(username_score, 0.4)  # Cap username contribution
        
        # Ensure the score doesn't exceed 1.0
        result['score'] = min(round(result['score'], 2), 1.0)
        return result
    
    def classify_severity(self, score: float) -> str:
        """Classify the severity based on the score"""
        if score >= self.rules.high_severity_threshold:
            return 'high'
        elif score >= self.rules.medium_severity_threshold:
            return 'medium'
        else:
            return 'low'
            
    def should_flag(self, score_details: Dict[str, Any]) -> bool:
        """
        Determine if content should be flagged as gambling
        Using both score and additional signals
        """
        score = score_details['score']
        
        # High score - definite flag
        if score >= self.rules.flagging_threshold:
            return True
            
        # Borderline score with supporting signals - flag
        if self.rules.borderline_check_threshold <= score < self.rules.flagging_threshold:
            if (score_details.get('has_promo', False) and 
                (score_details.get('has_wealth_emoji', False) or 
                 score_details.get('has_testimonial', False))):
                return True
                
        return False


class CommentProcessor:
    """Process a CSV file of comments to detect gambling content"""
    
    def __init__(self, rules: Optional[DetectionRules] = None):
        """Initialize with detection rules"""
        self.detector = GamblingDetector(rules)
        
    def process_csv(self, input_path: str, output_path: str, delimiter: str = '¤') -> None:
        """
        Process a CSV file of comments and output results
        
        Args:
            input_path: Path to the input CSV file
            output_path: Path to save the output CSV file
            delimiter: CSV delimiter character
        """
        try:
            # Read the input CSV
            logger.info(f"Reading input file: {input_path}")
            df = pd.read_csv(
                input_path, 
                header=None, 
                delimiter=delimiter, 
                encoding='utf-8-sig', 
                engine='python',
                on_bad_lines='warn'
            )
            
            # Assign column names
            if len(df.columns) >= 5:
                df.columns = ['videoId', 'username', 'comment', 'likes', 'publishedAt'] + \
                             [f'extra_{i}' for i in range(len(df.columns) - 5)]
            else:
                logger.warning(f"Expected at least 5 columns, got {len(df.columns)}. Using default column names.")
                default_cols = ['videoId', 'username', 'comment', 'likes', 'publishedAt']
                df.columns = default_cols[:len(df.columns)]
                
                # Add missing columns if needed
                for col in default_cols:
                    if col not in df.columns:
                        df[col] = None
            
            # Step 1: Score each comment and capture detailed results
            logger.info("Scoring comments...")
            
            # Process in batches for better performance monitoring
            total_rows = len(df)
            batch_size = 1000
            num_batches = (total_rows + batch_size - 1) // batch_size
            
            # Initialize new columns
            df['score'] = 0.0
            df['has_brand'] = False
            df['has_spam_pattern'] = False
            df['has_promo'] = False
            df['has_testimonial'] = False
            df['has_wealth_emoji'] = False
            df['username_score'] = 0.0
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_rows)
                logger.info(f"Processing batch {i+1}/{num_batches} (rows {start_idx}-{end_idx})...")
                
                batch = df.iloc[start_idx:end_idx]
                for idx, row in batch.iterrows():
                    try:
                        result = self.detector.score_comment(str(row['comment']), str(row['username']))
                        for key, value in result.items():
                            df.at[idx, key] = value
                    except Exception as e:
                        logger.error(f"Error processing row {idx}: {e}")
                        continue
            
            # Step 2: Mark severity
            df['severity'] = df['score'].apply(self.detector.classify_severity)
            
            # Step 3: Flag usernames that posted high-severity content
            high_severity_threshold = self.detector.rules.high_severity_threshold
            spam_users = set(df[df['score'] >= high_severity_threshold]['username'])
            df['username_flagged'] = df['username'].apply(lambda u: u in spam_users)
            
            # Step 4: Final score adjustment
            df['final_score'] = df.apply(
                lambda r: 1.0 if r['username_flagged'] else r['score'], 
                axis=1
            )
            df['final_severity'] = df['final_score'].apply(self.detector.classify_severity)
            
            # Step 5: Gambling flag logic
            df['gambling'] = df.apply(
                lambda r: self.detector.should_flag({
                    'score': r['final_score'],
                    'has_promo': r['has_promo'],
                    'has_wealth_emoji': r['has_wealth_emoji'],
                    'has_testimonial': r['has_testimonial']
                }),
                axis=1
            )
            
            # Calculate statistics
            flagged_count = df['gambling'].sum()
            flagged_percentage = (flagged_count / len(df)) * 100
            logger.info(f"Flagged {flagged_count} comments ({flagged_percentage:.2f}%)")
            
            by_severity = df['final_severity'].value_counts()
            logger.info(f"Severity distribution: {by_severity.to_dict()}")
            
            # Output
            logger.info(f"Saving processed data to {output_path}")
            df.to_csv(output_path, index=False, sep=delimiter, encoding='utf-8-sig')
            
            # Display sample results
            logger.info("Sample results:")
            sample_cols = ['comment', 'username', 'score', 'severity', 'final_score', 'final_severity', 'gambling']
            print(df[sample_cols].head(10))
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            raise


def main():
    """Main function to run the script"""
    
    # Create custom detection rules (optional)
    rules = DetectionRules(
        # These values use defaults from __post_init__
        high_severity_threshold=0.7,
        medium_severity_threshold=0.4,
        flagging_threshold=0.5,
        borderline_check_threshold=0.4
    )
    
    # Create processor with rules
    processor = CommentProcessor(rules)
    
    # Process the CSV file
    try:
        processor.process_csv("comments.csv", "comments_labeled.csv")
        logger.info("✅ Processing completed successfully")
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")


if __name__ == "__main__":
    main()
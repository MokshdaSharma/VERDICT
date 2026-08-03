import time
import threading
from typing import Callable, Any

class TokenBucket:
    def __init__(self, rpm: int, tpm: int):
        self.rpm = rpm
        self.tpm = tpm
        
        self.tokens_r = rpm
        self.tokens_t = tpm
        self.last_update = time.monotonic()
        self.lock = threading.Lock()
        
    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_update
        if elapsed > 0:
            self.tokens_r = min(self.rpm, self.tokens_r + elapsed * (self.rpm / 60.0))
            self.tokens_t = min(self.tpm, self.tokens_t + elapsed * (self.tpm / 60.0))
            self.last_update = now
            
    def consume(self, req_tokens: int = 0) -> float:
        with self.lock:
            self._refill()
            wait_time = 0.0
            
            # If we don't have enough RPM, calc wait
            if self.tokens_r < 1:
                wait_time = max(wait_time, (1 - self.tokens_r) / (self.rpm / 60.0))
                
            # If we don't have enough TPM, calc wait
            if self.tokens_t < req_tokens:
                wait_time = max(wait_time, (req_tokens - self.tokens_t) / (self.tpm / 60.0))
                
            if wait_time > 0:
                # Fast forward conceptually
                self.tokens_r += wait_time * (self.rpm / 60.0)
                self.tokens_t += wait_time * (self.tpm / 60.0)
                self.last_update += wait_time
                
            self.tokens_r -= 1
            self.tokens_t -= req_tokens
            
            return wait_time

def retry_with_backoff(func: Callable, max_retries: int = 5, base_delay: float = 1.0):
    def wrapper(*args, **kwargs):
        retries = 0
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check for 429 inside the exception message
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "Too Many Requests" in err_msg:
                    if retries >= max_retries:
                        raise e
                    
                    delay = base_delay * (2 ** retries)
                    
                    # Try to parse Retry-After from the exception if available (very basic attempt)
                    if "Retry-After" in err_msg or "retry in" in err_msg.lower():
                        # Simple heuristic, assumes format "retry in 59s"
                        import re
                        match = re.search(r"retry in ([\d\.]+)s", err_msg.lower())
                        if match:
                            delay = max(delay, float(match.group(1)))
                            
                    print(f"Rate limited (429). Retrying in {delay:.2f}s... (Attempt {retries+1}/{max_retries})")
                    time.sleep(delay)
                    retries += 1
                elif "503" in err_msg or "UNAVAILABLE" in err_msg:
                    if retries >= max_retries:
                        raise e
                    delay = base_delay * (2 ** retries)
                    print(f"Server overloaded (503). Retrying in {delay:.2f}s... (Attempt {retries+1}/{max_retries})")
                    time.sleep(delay)
                    retries += 1
                else:
                    raise e
    return wrapper

# Default Gemini 3.5 Flash free tier limits: 15 RPM, 1 million TPM (250k input?)
# Let's set it to 14 RPM, 200,000 TPM to be safe.
global_bucket = TokenBucket(rpm=14, tpm=200000)

def call_with_rate_limit(func, est_tokens: int, *args, **kwargs):
    wait_s = global_bucket.consume(req_tokens=est_tokens)
    if wait_s > 0:
        print(f"[RateLimiter] Sleeping {wait_s:.2f}s to respect token bucket...")
        time.sleep(wait_s)
    
    return retry_with_backoff(func)(*args, **kwargs)

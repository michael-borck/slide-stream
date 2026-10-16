# Hash Tables

- Find a value without searching
- Week 6, Data Structures

# A Hash Function

```python
# a comment, not a new slide
def bucket_for(key: str, n: int) -> int:
    return hash(key) % n
```

- `hash()` turns a key into an integer

# How Fast Is It?

| Operation | List | Hash table |
|---|---|---|
| Look up | O(n) | O(1) |
| Insert | O(1) | O(1) |

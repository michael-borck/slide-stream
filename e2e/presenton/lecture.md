# Hash Tables: Finding Things Fast

- How a dictionary finds a value without searching
- Week 6, Data Structures

# The Problem with Searching a List

- A list of 1,000,000 student records
- To find one ID, check each record in turn
- On average that is 500,000 comparisons
- Double the data, double the wait

# The Big Idea: Compute the Address

- Turn the key into a number with a hash function
- Use that number as an index into an array of buckets
- Go straight to the bucket instead of searching
- The cost no longer depends on how much data you have

# A Hash Function in Python

```python
def bucket_for(key: str, n_buckets: int) -> int:
    return hash(key) % n_buckets

bucket_for("s1234567", 8)   # -> a number from 0 to 7
```

- `hash()` turns any immutable value into an integer
- `%` folds it into the range of available buckets

# When Two Keys Collide

## Chaining

- Each bucket holds a small list
- Colliding keys share the bucket's list

## Open addressing

- On a collision, probe the next free bucket
- Python's `dict` uses a variant of this

# How Fast Is It?

| Operation | Unsorted list | Hash table (average) | Hash table (worst) |
|---|---|---|---|
| Look up a key | O(n) | O(1) | O(n) |
| Insert | O(1) | O(1) | O(n) |
| Delete | O(n) | O(1) | O(n) |

# Load Factor and Resizing

1. The load factor is the number of stored items divided by the number of buckets, and it is the single most useful figure for predicting how a hash table will behave under load.
2. As the load factor rises, collisions become more frequent, chains grow longer or probe sequences get longer, and the average lookup slowly drifts away from constant time.
3. Most implementations pick a threshold, often somewhere between 0.5 and 0.75, and grow the table when the load factor crosses it.
4. Growing means allocating a larger array, usually about double the size, and reinserting every existing item, because each key's bucket depends on the number of buckets.
5. A single resize is expensive, but it happens rarely enough that the cost spread across all inserts stays constant on average, which is called amortised constant time.

# Where You Already Use Them

- Python `dict` and `set`
- Database indexes and caches
- Git object storage, keyed by content hash
- Compilers' symbol tables

# Key Takeaways

- Hashing trades memory for speed
- Average lookup is O(1); the worst case is O(n)
- Keep the load factor low by resizing
- Keys must be immutable, so their hash never changes

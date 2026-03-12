function maxs(arr, n, k) {
    
    // Initialize result
    let max_s = Number.MIN_SAFE_INTEGER;

    // Consider all blocks starting with i
    for (let i = 0; i < n - k + 1; i++) {
        let current_s = 0;
        for (let j = 0; j < k; j++) {
            current_s += arr[i + j];
        }

        // Update result if required
        max_s = Math.max(current_s, max_s);
    }

    return max_s;
}

// Driver code
const arr = [5, 2, -1, 0, 3];
const k = 3;
const n = arr.length;
console.log(maxs(arr, n, k));
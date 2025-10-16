# ===============================
# Rare Variant Detection: FM + DGIM + Exact MapReduce
# Adapted for no-header TSV dataset
# ===============================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, when
import hashlib

# -------------------------------
# 1️⃣ Initialize Spark
# -------------------------------
spark = SparkSession.builder \
    .appName("RareVariantDetection") \
    .config("spark.sql.execution.arrow.enabled", "true") \
    .getOrCreate()

# -------------------------------
# 2️⃣ Load Dataset (no header)
# -------------------------------
data_path = "/user/cloudera/exac_variants.tsv"

df = spark.read.csv(
    data_path,
    sep="\t",
    header=False,
    inferSchema=True
)

print("✅ Dataset Loaded Successfully!")
df.show(5)
df.printSchema()

# -------------------------------
# 3️⃣ Create Variant ID and Rare Indicator
# -------------------------------
# variant_id = chromosome + position + ref + alt
df = df.withColumn("variant_id", concat_ws("_", col("_c0"), col("_c1"), col("_c2"), col("_c3")))

# is_rare = 1 if allele frequency (_c7) < 0.01, else 0
df = df.withColumn("is_rare", when(col("_c7") < 0.01, 1).otherwise(0))

# -------------------------------
# 4️⃣ Flajolet-Martin (FM) Class
# -------------------------------
class FlajoletMartin:
    def __init__(self, num_hashes=32):
        self.num_hashes = num_hashes
        self.max_trailing_zeros = [0] * num_hashes

    def _hash(self, value, seed):
        h = hashlib.sha1(f"{seed}_{value}".encode("utf-8")).hexdigest()
        return int(h, 16)

    def _trailing_zeros(self, x):
        if x == 0:
            return 0
        tz = 0
        while x & 1 == 0:
            x >>= 1
            tz += 1
        return tz

    def add(self, value):
        for i in range(self.num_hashes):
            h = self._hash(value, i)
            tz = self._trailing_zeros(h)
            self.max_trailing_zeros[i] = max(self.max_trailing_zeros[i], tz)

    def estimate(self):
        avg = sum([2 ** x for x in self.max_trailing_zeros]) / self.num_hashes
        return int(avg)

# -------------------------------
# 5️⃣ DGIM (Sliding Window) Class
# -------------------------------
class DGIM:
    def __init__(self, window_size):
        self.window_size = window_size
        self.buckets = []

    def _current_time(self):
        return len(self.buckets)

    def add_bit(self, bit):
        t = self._current_time()
        if bit == 1:
            self.buckets.insert(0, (t, 1))
            self._compress()

        while self.buckets and self.buckets[-1][0] <= t - self.window_size:
            self.buckets.pop()

    def _compress(self):
        i = 0
        while i < len(self.buckets) - 2:
            if (self.buckets[i][1] == self.buckets[i + 1][1] ==
                    self.buckets[i + 2][1]):
                merged = (self.buckets[i + 1][0], self.buckets[i + 1][1] * 2)
                self.buckets = self.buckets[:i + 1] + [merged] + self.buckets[i + 3:]
            else:
                i += 1

    def query(self):
        if not self.buckets:
            return 0
        total = sum([b[1] for b in self.buckets[:-1]])
        total += self.buckets[-1][1] // 2
        return total

# -------------------------------
# 6️⃣ Initialize FM and DGIM
# -------------------------------
fm = FlajoletMartin(num_hashes=32)
window_size = 1000
dgim = DGIM(window_size=window_size)

# -------------------------------
# 7️⃣ Process Variants
# -------------------------------
variant_col = "variant_id"
rare_col = "is_rare"

rare_counts = []
variants = df.select(variant_col, rare_col).collect()

for row in variants:
    variant = row[variant_col]
    is_rare = row[rare_col]

    # FM: estimate distinct rare variants
    if is_rare == 1:
        fm.add(str(variant))

    # DGIM: track rare variant frequency
    dgim.add_bit(is_rare)
    rare_counts.append(dgim.query())

# -------------------------------
# 8️⃣ MapReduce Exact Count
# -------------------------------
exact_count = df.filter(col(rare_col) == 1).select(variant_col).distinct().count()

exact_window_counts = []
bits = [row[rare_col] for row in variants]
for i in range(window_size, len(bits)):
    exact_window_counts.append(sum(bits[i - window_size:i]))

# -------------------------------
# 9️⃣ Print Results
# -------------------------------
print("\n==============================")
print("🔍 Rare Variant Detection Results")
print("==============================")
print("✅ FM Estimated Distinct Rare Variants:", fm.estimate())
print("✅ Exact Distinct Rare Variants:", exact_count)
print("✅ DGIM Sliding Window Counts (last 10):", rare_counts[-10:])
print("✅ Exact Sliding Window Counts (last 10):", exact_window_counts[-10:])
print("==============================\n")

spark.stop()

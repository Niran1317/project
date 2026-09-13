import java.util.*;

public class Main {
    static class Graph {
        private final Map<Integer, Set<Integer>> adjacency = new HashMap<>();

        void addEdge(int a, int b) {
            adjacency.computeIfAbsent(a, k -> new HashSet<>()).add(b);
            adjacency.computeIfAbsent(b, k -> new HashSet<>()).add(a);
        }

        Set<Integer> nodes() {
            return adjacency.keySet();
        }

        Set<Integer> neighbors(int node) {
            return adjacency.getOrDefault(node, Collections.emptySet());
        }

        int degree(int node) {
            return neighbors(node).size();
        }

        Set<Integer> greedyBlock(int k) {
            Set<Integer> blocked = new LinkedHashSet<>();
            Set<Integer> candidates = new HashSet<>(nodes());

            for (int i = 0; i < k && !candidates.isEmpty(); i++) {
                int best = -1;
                int bestDegree = -1;

                for (int node : candidates) {
                    int score = 0;
                    for (int n : neighbors(node)) {
                        if (!blocked.contains(n)) score++;
                    }
                    if (score > bestDegree) {
                        bestDegree = score;
                        best = node;
                    }
                }

                if (best != -1) {
                    blocked.add(best);
                    candidates.remove(best);
                }
            }
            return blocked;
        }
    }

    static class BloomFilter {
        private final BitSet bits;
        private final int size;

        BloomFilter(int size) {
            this.size = size;
            this.bits = new BitSet(size);
        }

        private int hash(String text, int seed) {
            int h = seed;
            for (char c : text.toLowerCase(Locale.ROOT).toCharArray()) {
                h = 31 * h + c;
            }
            return Math.floorMod(h, size);
        }

        void add(String text) {
            bits.set(hash(text, 17));
            bits.set(hash(text, 31));
            bits.set(hash(text, 73));
        }

        boolean mightContain(String text) {
            return bits.get(hash(text, 17))
                    && bits.get(hash(text, 31))
                    && bits.get(hash(text, 73));
        }
    }

    static boolean isSpam(String message, BloomFilter filter) {
        String[] spamWords = {"free", "winner", "prize", "click", "urgent"};
        for (String word : spamWords) {
            if (filter.mightContain(word) && message.toLowerCase(Locale.ROOT).contains(word)) {
                return true;
            }
        }
        return false;
    }

    public static void main(String[] args) {
        Graph graph = new Graph();

        int[][] edges = {
            {1,2},{1,3},{1,4},{2,3},{2,5},
            {3,4},{3,5},{4,6},{5,6},{5,7},
            {6,7},{7,8},{6,8}
        };

        for (int[] edge : edges) {
            graph.addEdge(edge[0], edge[1]);
        }

        Set<Integer> blocked = graph.greedyBlock(2);

        System.out.println("=== Misinformation Influence Minimization ===");
        System.out.println("Nodes: " + graph.nodes());
        System.out.println("Degrees:");
        for (int node : new TreeSet<>(graph.nodes())) {
            System.out.println("Node " + node + " -> " + graph.degree(node));
        }

        System.out.println("\nGreedy blocked nodes: " + blocked);

        BloomFilter filter = new BloomFilter(256);
        for (String word : new String[]{"free", "winner", "prize", "click", "urgent"}) {
            filter.add(word);
        }

        String[] messages = {
            "Urgent! Click now to claim your free prize.",
            "Meeting moved to 3 PM today."
        };

        System.out.println("\nSpam detection:");
        for (String message : messages) {
            System.out.println(message + " -> " + (isSpam(message, filter) ? "SPAM" : "NOT SPAM"));
        }
    }
}

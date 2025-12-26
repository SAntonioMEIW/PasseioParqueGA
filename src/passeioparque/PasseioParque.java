/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Main.java to edit this template
 */
package passeioparque;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
/**
 *
 * @author Silvia
 */
/**
 * E-fólio B — Introdução à IA — Algoritmo Evolutivo (GA) para melhor passeio e orientação
 *
 * Indivíduo: trajeto (lista de (x,y))
 * Avaliação: satisfação conforme regras e custo = (T + K) - satisfação, fitness = -custo
 * Paragem: 1_000_000 avaliações ou 10 segundos
 */

public class PasseioParque {

    // -----------------------------
    // Representação do Parque
    // -----------------------------
    static class Park {
        final int N;              // tamanho do lado
        final int[][] grid;       // valores (1,2,10,-poi)
        final int T;              // tempo máximo do passeio
        final int K;              // soma dos pontos de interesse
        final List<int[]> doors;  // coordenadas das portas (N,S,E,W)

        Park(int[][] grid, int T, int K) {
            this.grid = grid;
            this.N = grid.length;
            this.T = T;
            this.K = K;
            this.doors = computeDoors();
        }

        //representa um acesso válido no parque
        boolean inside(int x, int y) {
            return x >= 0 && x < N && y >= 0 && y < N;
        }

        //representa um acesso inválido no parque
        boolean inaccessible(int x, int y) {
            return grid[x][y] == 10;
        }

        //custo do percurso
        int stepCost(int x, int y) {
            int cell = grid[x][y];
            return (cell == 2) ? 2 : 1; // 2 para ":" e 1 para "." e pontos
        }

        //ponto de interesse
        int poiValue(int x, int y) {
            int cell = grid[x][y];
            return cell < 0 ? -cell : 0;
        }

        // Portas: no meio de cada lado
        private List<int[]> computeDoors() {
            List<int[]> ds = new ArrayList<>();
            int mid = N / 2;
            // Norte: (0, mid)
            if (!inaccessible(0, mid)) ds.add(new int[]{0, mid});
            // Sul: (N-1, mid)
            if (!inaccessible(N - 1, mid)) ds.add(new int[]{N - 1, mid});
            // Oeste: (mid, 0)
            if (!inaccessible(mid, 0)) ds.add(new int[]{mid, 0});
            // Este: (mid, N-1)
            if (!inaccessible(mid, N - 1)) ds.add(new int[]{mid, N - 1});
            return ds;
        }
    }

    // -----------------------------
    // Mapa de distâncias às portas
    // -----------------------------
    static class DistanceMap {
        final int[][] dist; // minutos até sair por qualquer porta (1 à frente da porta)
        static final int INF = 1_000_000_000;

        DistanceMap(Park park) {
            this.dist = computeDistances(park);
        }

        private int[][] computeDistances(Park park) {
            int N = park.N;
            int[][] d = new int[N][N];
            for (int i = 0; i < N; i++) Arrays.fill(d[i], INF);

            // BFS multi-fonte com custo uniforme por passo; tratamento especial para portas
            Deque<int[]> q = new ArrayDeque<>();
            for (int[] door : park.doors) {
                int x = door[0], y = door[1];
                d[x][y] = 1; // "em frente à porta" distância 1 para sair
                q.add(new int[]{x, y});
            }

            int[] dx = {-1, 1, 0, 0};
            int[] dy = {0, 0, -1, 1};

            while (!q.isEmpty()) {
                int[] cur = q.poll();
                int x = cur[0], y = cur[1];
                for (int k = 0; k < 4; k++) {
                    int nx = x + dx[k], ny = y + dy[k];
                    if (!park.inside(nx, ny) || park.inaccessible(nx, ny)) continue;
                    int cost = park.stepCost(nx, ny); // custo de entrar na célula vizinha
                    int nd = d[x][y] + cost;
                    if (nd < d[nx][ny]) {
                        d[nx][ny] = nd;
                        q.add(new int[]{nx, ny});
                    }
                }
            }
            return d;
        }
    }

    // -----------------------------
    // Indivíduo (trajeto) e avaliação
    // -----------------------------
    static class Path {
        final Park park;
        final List<int[]> steps = new ArrayList<>(); // sequência de casas (x,y)

        Path(Park park) {
            this.park = park;
        }

        Path clonePath() {
            Path p = new Path(park);
            for (int[] st : steps) p.steps.add(new int[]{st[0], st[1]});
            return p;
        }

        // Avaliação conforme regras do enunciado
        EvalResult evaluate() {
            int time = 0;
            int satisf = 0;
            Set<Long> visited = new HashSet<>();
            int consecutiveRevisits = 0;
            for (int[] st : steps) {
                int x = st[0], y = st[1];
                
                //Verifica se o passo é válido (dentro do parque e acessível)
                if (!park.inside(x, y) || park.inaccessible(x, y)) break;
                int cost = park.stepCost(x, y); //custo do passo
                time += cost;
                if (time > park.T) break;

                long key = (((long) x) << 32) ^ (long) y;
                boolean isNew = visited.add(key);

                if (isNew) {
                    satisf += 1; // bônus por novidade
                    consecutiveRevisits = 0;
                    int poi = park.poiValue(x, y);
                    if (poi > 0) satisf += poi;
                } else {
                    consecutiveRevisits++;
                    if (consecutiveRevisits > 1) satisf -= 1; // desconto a partir da segunda revisita consecutiva
                }
            }
            int ideal = park.T + park.K;
            int cost = ideal - satisf;
            int fitness = -cost;
            return new EvalResult(satisf, cost, fitness, time);
        }
    }

    static class EvalResult {
        final int satisfaction;
        final int cost;
        final int fitness;
        final int timeUsed;

        EvalResult(int satisfaction, int cost, int fitness, int timeUsed) {
            this.satisfaction = satisfaction;
            this.cost = cost;
            this.fitness = fitness;
            this.timeUsed = timeUsed;
        }
    }

    // -----------------------------
    // Algoritmo Genético (GA)
    // -----------------------------
    static class GA {
        final Park park;
        final DistanceMap dmap;
        final int popSize;
        final int maxEvaluations;
        final long maxMillis;
        final double pmutate;
        final double pcross;

        long evaluations = 0;
        long expansions = 0; // opcional: vizinhanças ou mutações aplicadas
        int generations = 0;

        GA(Park park, DistanceMap dmap, int popSize, int maxEvaluations, long maxMillis,
           double pmutate, double pcross) {
            this.park = park;
            this.dmap = dmap;
            this.popSize = popSize;
            this.maxEvaluations = maxEvaluations;
            this.maxMillis = maxMillis;
            this.pmutate = pmutate;
            this.pcross = pcross;
        }

        Result run() {
            long start = System.currentTimeMillis();
            List<Path> population = initPopulation();
            Path best = null;
            EvalResult bestEval = null;

            while (true) {
                // Avaliação
                List<EvalResult> evals = new ArrayList<>(population.size());
                for (Path p : population) {
                    EvalResult er = p.evaluate();
                    evaluations++;
                    evals.add(er);
                    if (best == null || er.fitness > bestEval.fitness) {
                        best = p.clonePath();
                        bestEval = er;
                    }
                }

                generations++;

                // Critérios de paragem
                long elapsed = System.currentTimeMillis() - start;
                if (evaluations >= maxEvaluations || elapsed >= maxMillis) {
                    return new Result(best, bestEval, evaluations, generations, expansions, elapsed / 1000.0);
                }

                // Seleção por torneio
                List<Path> matingPool = new ArrayList<>(popSize);
                for (int i = 0; i < popSize; i++) {
                    int a = randInt(popSize), b = randInt(popSize);
                    Path winner = evals.get(a).fitness >= evals.get(b).fitness ? population.get(a) : population.get(b);
                    matingPool.add(winner.clonePath());
                }

                // Cruzamento
                List<Path> offspring = new ArrayList<>(popSize);
                for (int i = 0; i < popSize; i += 2) {
                    Path p1 = matingPool.get(i);
                    Path p2 = matingPool.get((i + 1) % popSize);
                    if (randDouble() < pcross) {
                        Path[] kids = crossoverOnePoint(p1, p2);
                        offspring.add(kids[0]);
                        offspring.add(kids[1]);
                    } else {
                        offspring.add(p1.clonePath());
                        offspring.add(p2.clonePath());
                    }
                }
                if (offspring.size() > popSize) offspring.remove(offspring.size() - 1);

                // Mutação
                for (Path child : offspring) {
                    if (randDouble() < pmutate) {
                        mutateStep(child);
                        expansions++;
                    }
                }

                // Elitismo: garante que o melhor atual permanece
                int worstIdx = 0;
                int worstFit = Integer.MAX_VALUE;
                for (int i = 0; i < offspring.size(); i++) {
                    int fit = offspring.get(i).evaluate().fitness; // rápida checagem
                    if (fit < worstFit) {
                        worstFit = fit;
                        worstIdx = i;
                    }
                }
                offspring.set(worstIdx, best.clonePath());

                population = offspring;
            }
        }

        // Inicialização: trajetos que respeitam tempo T e evitam células inacessíveis
        private List<Path> initPopulation() {
            List<Path> pop = new ArrayList<>(popSize);
            for (int i = 0; i < popSize; i++) {
                Path p = new Path(park);
                randomFeasiblePath(p);
                pop.add(p);
            }
            return pop;
        }

        // Geração de caminho aleatório viável
        private void randomFeasiblePath(Path p) {
            p.steps.clear();
            int N = park.N;
            int trials = 0;

            // escolhe uma posição inicial acessível
            int x, y;
            do {
                x = randInt(N);
                y = randInt(N);
                trials++;
            } while ((park.inaccessible(x, y)) && trials < N * N);

            int time = 0;
            p.steps.add(new int[]{x, y});

            int[] dx = {-1, 1, 0, 0};
            int[] dy = {0, 0, -1, 1};

            // caminho guiado pelas distâncias para favorecer saída e diversidade
            while (time < park.T) {
                List<int[]> candidates = new ArrayList<>(4);
                for (int k = 0; k < 4; k++) {
                    int nx = x + dx[k], ny = y + dy[k];
                    if (!park.inside(nx, ny) || park.inaccessible(nx, ny)) continue;
                    int stepCost = park.stepCost(nx, ny);
                    if (time + stepCost > park.T) continue;
                    candidates.add(new int[]{nx, ny});
                }
                if (candidates.isEmpty()) break;

                // opção: escolhe uma célula com menor distância à porta, com ruído
                candidates.sort(Comparator.comparingInt(a -> dmap.dist[a[0]][a[1]] + randInt(3)));
                int[] chosen = candidates.get(0);
                x = chosen[0]; y = chosen[1];
                p.steps.add(new int[]{x, y});
                time += park.stepCost(x, y);
            }
        }

        // Cruzamento 1 ponto com alinhamento por comprimento
        private Path[] crossoverOnePoint(Path a, Path b) {
            Path c1 = a.clonePath(), c2 = b.clonePath();
            int len = Math.min(a.steps.size(), b.steps.size());
            if (len <= 1) return new Path[]{c1, c2};
            int cut = 1 + randInt(len - 1); // evita 0 e len
            c1.steps.clear();
            c2.steps.clear();
            // c1 = a[0..cut) + b[cut..end)
            c1.steps.addAll(copySlice(a.steps, 0, cut));
            c1.steps.addAll(copySlice(b.steps, cut, b.steps.size()));
            // c2 = b[0..cut) + a[cut..end)
            c2.steps.addAll(copySlice(b.steps, 0, cut));
            c2.steps.addAll(copySlice(a.steps, cut, a.steps.size()));
            // repara viabilidade (limites/inacessíveis/tempo)
            repairPath(c1);
            repairPath(c2);
            return new Path[]{c1, c2};
        }

        private List<int[]> copySlice(List<int[]> src, int s, int e) {
            List<int[]> out = new ArrayList<>(Math.max(0, e - s));
            for (int i = s; i < e; i++) out.add(new int[]{src.get(i)[0], src.get(i)[1]});
            return out;
        }

        // Mutação: desloca um ponto em uma das 4 direções ou faz inserção/remoção pequena
        private void mutateStep(Path p) {
            if (p.steps.isEmpty()) {
                randomFeasiblePath(p);
                return;
            }
            int choice = randInt(3);
            if (choice == 0) {
                // desloca um índice
                int idx = randInt(p.steps.size());
                int[] st = p.steps.get(idx);
                int[][] deltas = {{-1,0},{1,0},{0,-1},{0,1}};
                int[] d = deltas[randInt(4)];
                int nx = st[0] + d[0], ny = st[1] + d[1];
                if (park.inside(nx, ny) && !park.inaccessible(nx, ny)) {
                    p.steps.set(idx, new int[]{nx, ny});
                    repairPath(p);
                }
            } else if (choice == 1) {
                // remove um passo se houver mais de 1
                if (p.steps.size() > 1) {
                    p.steps.remove(randInt(p.steps.size()));
                    repairPath(p);
                }
            } else {
                // insere um passo viável próximo ao fim
                int[] last = p.steps.get(p.steps.size() - 1);
                List<int[]> cand = neighbors(last[0], last[1]);
                if (!cand.isEmpty()) {
                    int[] c = cand.get(randInt(cand.size()));
                    p.steps.add(new int[]{c[0], c[1]});
                    repairPath(p);
                }
            }
        }

        private List<int[]> neighbors(int x, int y) {
            List<int[]> out = new ArrayList<>(4);
            int[] dx = {-1, 1, 0, 0};
            int[] dy = {0, 0, -1, 1};
            for (int k = 0; k < 4; k++) {
                int nx = x + dx[k], ny = y + dy[k];
                if (park.inside(nx, ny) && !park.inaccessible(nx, ny)) out.add(new int[]{nx, ny});
            }
            return out;
        }

        // Reparação: corta fora passos inválidos e respeita tempo T
        private void repairPath(Path p) {
            List<int[]> repaired = new ArrayList<>(p.steps.size());
            int time = 0;
            for (int[] st : p.steps) {
                int x = st[0], y = st[1];
                if (!park.inside(x, y) || park.inaccessible(x, y)) break;
                int cost = park.stepCost(x, y);
                if (time + cost > park.T) break;
                repaired.add(new int[]{x, y});
                time += cost;
            }
            if (repaired.isEmpty()) {
                randomFeasiblePath(p);
            } else {
                p.steps.clear();
                p.steps.addAll(repaired);
            }
        }

        private int randInt(int bound) {
            return ThreadLocalRandom.current().nextInt(bound);
        }

        private double randDouble() {
            return ThreadLocalRandom.current().nextDouble();
        }
    }

    static class Result {
        final Path best;
        final EvalResult eval;
        final long evaluations;
        final int generations;
        final long expansions;
        final double seconds;

        Result(Path best, EvalResult eval, long evaluations, int generations, long expansions, double seconds) {
            this.best = best;
            this.eval = eval;
            this.evaluations = evaluations;
            this.generations = generations;
            this.expansions = expansions;
            this.seconds = seconds;
        }
    }

    // -----------------------------
    // Instâncias de exemplo (minimais)
    // -----------------------------
    static class Instances {
        static List<Park> load() {
            List<Park> list = new ArrayList<>();

            // Instância 1: N=5, K=6, W=4, T=10 (do enunciado base)
            int[][] g1 = {
                    {1, 1, 1, 1, -1},
                    {1, 10, 1, 2, 10},
                    {1, -2, 10, 2, 1},
                    {10, 10, 1, 1, 1},
                    {-1, 1, 1, 2, -2}
            };
            list.add(new Park(g1, 10, 6));

            // Instância 2: N=5, K=6, W=4, T=20 (exemplo alternativo)
            int[][] g2 = {
                    {1, 1, 1, 1, -1},
                    {1, 1, 1, 2, 1},
                    {10, -2, 2, 1, 1},
                    {1, 1, 10, 1, 1},
                    {-1, 1, 1, 2, -2}
            };
            list.add(new Park(g2, 20, 6));

            // Instância 3: N=7, K=5, W=4, T=20 (exemplo alternativo)
            int[][] g3 = {
                    {1, 1, 1, 1, 10, -2, 1},
                    {1, -2, 10, 1, 1, 10, 1},
                    {1, 10, 1, 10, 1, 1, 1},
                    {1, 1, -2, 10, 1, 2, 1},
                    {2, 1, 10, 1, 2, 1, 10},
                    {2, 1, 1, 1, 2, 10, -3},
                    {1, -1, 10, 1, 1, 1, 1}
            };
            list.add(new Park(g3, 10, 10));

            // Instância 3: N=15, K=38, W=15, T=30 (exemplo alternativo)
            int[][] g4 = {
                    {-3, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, -3, 1, -2},
                    {1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 1, 1, 2, 1, 2},
                    {1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, -2},
                    {1, 1, 1, 1, 1, 1, -2, 2, 1, -3, 1, 1, 1, 1, 1},
                    {1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1},
                    {1, 10, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, -1, 10, 1},
                    {1, 10, -3, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 10, 1},
                    {1, 10, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 10, 1},
                    {1, 10, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 10, 2},
                    {1, 10, -2, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 10, 1},
                    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2},
                    {-2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, -3, 1},
                    {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1},
                    {1, 2, 1, 1, 1, 10, 10, 10, 10, 10, 1, 1, 2, 1, 1},
                    {1, 1, -3, 1, 1, 1, 1, 1, 1, 1, 1, -3, 1, 1, -4}
            };
            list.add(new Park(g4, 30, 39));

            // Você pode adicionar as restantes instâncias conforme a tabela do PDF.
            return list;
        }
    }

    // -----------------------------
    // Saída da tabela de resultados
    // -----------------------------
    static void printResults(List<Result> results) {
        System.out.println("Instância\tCusto(g)\tExpansões\tGerações\tAvaliações\tTempo(s)");
        int idx = 1;
        int totalCost = 0;
        long totalExp = 0, totalEval = 0;
        int totalGen = 0;
        double totalSec = 0.0;

        for (Result r : results) {
            System.out.printf(Locale.US, "%d\t\t%d\t\t%d\t\t%d\t\t%d\t\t%.3f%n",
                    idx, r.eval.cost, r.expansions, r.generations, r.evaluations, r.seconds);
            totalCost += r.eval.cost;
            totalExp += r.expansions;
            totalEval += r.evaluations;
            totalGen += r.generations;
            totalSec += r.seconds;
            idx++;
        }
        System.out.printf(Locale.US, "Total:\t\t%d\t\t%d\t\t%d\t\t%d\t\t%.3f%n",
                totalCost, totalExp, totalGen, totalEval, totalSec);
    }

    // -----------------------------
    // Impressão da distância às portas (orientação)
    // -----------------------------
    static void printDistanceMap(Park park, DistanceMap dm) {
        System.out.println("Distâncias às portas (minutos):");
        for (int i = 0; i < park.N; i++) {
            for (int j = 0; j < park.N; j++) {
                int v = dm.dist[i][j];
                String s = (v >= DistanceMap.INF / 2) ? "#" : Integer.toString(v);
                System.out.print(s + "\t");
            }
            System.out.println();
        }
        System.out.println();
    }

    // -----------------------------
    // Execução principal
    // -----------------------------
    public static void main(String[] args) {
        List<Park> parks = Instances.load();
        List<Result> results = new ArrayList<>();

        // Configuração base do GA (ajuste conforme necessário)        
        int popSize = 150; //população
        int maxEvaluations = 1_000_000; 
        long maxMillis = 10_000; // 10 segundos
        double pmutate = 0.25; //taxa de mutação
        double pcross = 0.9; // taxa de cruzamento


        int instIdx = 1;
        for (Park park : parks) {
            System.out.println("=== Instância " + instIdx + " (N=" + park.N + ", T=" + park.T + ", K=" + park.K + ") ===");
            DistanceMap dmap = new DistanceMap(park);
            printDistanceMap(park, dmap);

            GA ga = new GA(park, dmap, popSize, maxEvaluations, maxMillis, pmutate, pcross);
            Result r = ga.run();
            results.add(r);

            // Mostrar melhor trajeto
            System.out.println("Melhor satisfação: " + r.eval.satisfaction + ", custo: " + r.eval.cost + ", tempo: " + r.eval.timeUsed);
            System.out.println("Trajeto (x,y):");
            for (int[] st : r.best.steps) System.out.print("(" + st[0] + "," + st[1] + ") ");
            System.out.println("\n");
            instIdx++;
        }

        // Tabela final
        printResults(results);
    }
}

    


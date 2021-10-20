import javafx.util.Pair;

import java.util.*;
import java.util.stream.Collectors;


class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        Lc126 lc126 = new Lc126();
        long timing = System.currentTimeMillis();

//        System.out.println(s.rearrangeString("cijfgcdadbhffgjdjccbdgeihbdcbjdijajehgihihdijghcbffiedjahbdjbbjcfggaj", 6));
        System.out.println(s.distanceLimitedPathsExist(277,
                new int[][]{{104, 197, 42}, {150, 265, 641}, {241, 160, 282}, {87, 59, 721}, {11, 71, 773}, {2, 256, 107}, {72, 7, 151}, {258, 163, 859}, {168, 241, 423}, {263, 1, 128}, {238, 8, 33}, {127, 101, 619}, {270, 46, 767}, {186, 52, 951}, {257, 122, 217}, {173, 232, 641}, {126, 150, 316}, {109, 151, 946}, {184, 175, 495}, {78, 56, 797}, {205, 240, 296}, {51, 37, 586}, {161, 17, 377}, {242, 106, 453}, {154, 162, 54}, {73, 178, 323}, {276, 229, 135}, {62, 65, 81}, {232, 231, 51}, {94, 229, 523}, {77, 267, 24}, {12, 7, 96}, {55, 192, 455}, {59, 251, 515}, {76, 152, 516}, {125, 174, 112}, {239, 143, 350}, {193, 46, 397}, {219, 15, 661}, {117, 251, 191}, {0, 220, 379}, {125, 87, 516}, {219, 252, 671}, {14, 96, 354}, {3, 111, 684}, {70, 75, 846}, {38, 175, 652}, {222, 255, 381}, {49, 26, 850}, {57, 94, 279}, {82, 83, 633}, {173, 244, 820}, {264, 15, 772}, {29, 38, 476}, {164, 13, 483}, {118, 266, 684}, {214, 52, 24}, {52, 89, 747}, {102, 7, 786}, {8, 138, 156}, {212, 200, 490}, {107, 118, 562}, {46, 56, 885}, {30, 273, 587}, {28, 212, 998}, {38, 53, 814}, {97, 106, 985}, {47, 1, 428}, {0, 77, 941}, {111, 116, 856}, {130, 4, 329}, {48, 85, 523}, {188, 139, 211}, {80, 58, 415}, {79, 218, 959}, {66, 136, 269}, {184, 114, 881}, {135, 53, 125}, {88, 0, 84}, {6, 260, 964}, {89, 183, 389}, {38, 83, 74}, {47, 207, 89}, {135, 20, 669}, {50, 93, 118}, {55, 134, 230}, {182, 224, 220}, {176, 242, 616}, {50, 214, 436}, {208, 73, 504}, {13, 50, 448}, {18, 67, 704}, {93, 225, 642}, {128, 99, 281}, {4, 158, 89}, {274, 174, 448}, {253, 230, 244}, {115, 170, 500}, {142, 221, 851}, {181, 182, 220}, {235, 226, 981}, {85, 229, 420}, {163, 44, 485}, {217, 247, 291}, {241, 82, 250}, {220, 237, 375}, {72, 270, 299}, {39, 229, 36}, {34, 263, 828}, {150, 65, 25}, {232, 234, 118}, {122, 243, 356}, {244, 181, 161}, {275, 255, 666}, {261, 174, 731}, {258, 23, 108}, {37, 83, 906}, {272, 55, 998}, {250, 173, 219}, {25, 164, 952}, {174, 153, 258}, {192, 209, 314}, {37, 249, 748}, {61, 198, 747}, {28, 63, 503}, {117, 23, 67}, {218, 11, 779}, {276, 172, 422}, {88, 40, 661}, {82, 60, 283}, {80, 183, 37}, {0, 246, 217}, {101, 178, 102}, {233, 18, 716}, {38, 98, 698}, {269, 137, 196}, {85, 237, 562}, {198, 15, 918}, {276, 258, 264}, {45, 29, 218}, {217, 244, 663}, {224, 9, 673}, {91, 43, 901}, {265, 79, 669}, {11, 191, 467}, {191, 114, 834}, {142, 226, 914}, {224, 139, 994}, {228, 17, 176}, {146, 44, 886}, {31, 51, 577}, {157, 239, 847}, {257, 151, 330}, {64, 90, 420}, {218, 127, 397}, {112, 98, 531}, {149, 176, 163}, {55, 176, 744}, {37, 239, 852}, {158, 19, 801}, {133, 177, 457}, {24, 168, 240}, {198, 248, 368}, {215, 17, 434}, {71, 68, 433}, {132, 149, 656}, {9, 250, 940}, {224, 272, 56}, {257, 152, 267}, {99, 239, 898}, {74, 12, 562}, {133, 150, 784}, {19, 83, 559}, {205, 170, 647}, {176, 64, 975}, {101, 249, 115}, {77, 247, 127}, {60, 226, 278}, {221, 204, 736}, {119, 134, 417}, {165, 158, 702}, {179, 105, 327}, {241, 206, 6}, {97, 90, 720}, {50, 182, 992}, {219, 220, 228}, {53, 70, 493}, {75, 137, 465}, {16, 40, 362}, {184, 39, 980}, {99, 76, 61}, {219, 90, 934}, {151, 119, 95}, {139, 147, 373}, {256, 80, 100}, {164, 268, 880}, {112, 189, 177}, {82, 159, 263}, {148, 179, 127}, {210, 214, 910}, {233, 167, 708}, {259, 157, 796}, {236, 74, 890}, {118, 126, 217}, {65, 55, 304}, {42, 56, 984}, {218, 205, 800}, {114, 271, 691}, {61, 51, 923}, {109, 106, 432}, {142, 178, 429}, {29, 95, 51}, {248, 227, 940}, {90, 97, 135}, {19, 127, 954}, {24, 160, 308}, {74, 195, 306}, {81, 8, 925}, {90, 4, 599}, {118, 141, 602}, {268, 116, 663}, {37, 227, 904}, {276, 242, 267}, {234, 239, 691}, {105, 217, 287}, {61, 191, 257}, {119, 48, 993}, {55, 97, 260}, {50, 158, 380}, {156, 15, 255}, {275, 151, 579}, {42, 230, 579}, {49, 70, 250}, {190, 151, 445}, {55, 271, 732}, {239, 206, 546}, {266, 190, 783}, {64, 56, 79}, {77, 146, 908}, {159, 127, 332}, {33, 204, 152}, {235, 241, 680}, {127, 95, 810}, {117, 65, 894}, {199, 109, 411}, {117, 257, 828}, {78, 88, 126}, {65, 10, 967}, {144, 11, 81}, {19, 174, 778}, {170, 3, 944}, {101, 196, 281}, {98, 107, 75}, {26, 117, 510}, {239, 39, 589}, {195, 235, 863}, {222, 274, 923}, {206, 36, 861}, {238, 35, 168}, {115, 152, 498}, {19, 87, 672}, {17, 172, 951}, {110, 23, 264}, {72, 136, 844}, {260, 264, 269}, {236, 255, 219}, {18, 202, 26}, {24, 211, 568}, {218, 208, 247}, {153, 220, 32}, {148, 255, 565}, {196, 6, 192}, {207, 236, 553}, {171, 93, 773}, {233, 117, 424}, {175, 256, 986}, {74, 49, 435}, {116, 206, 660}, {131, 218, 664}, {271, 142, 158}, {14, 84, 348}, {75, 68, 719}, {18, 184, 820}, {85, 159, 433}, {50, 22, 204}, {196, 233, 859}, {178, 12, 531}, {220, 54, 815}, {188, 56, 361}, {190, 104, 605}, {188, 83, 141}, {169, 127, 596}, {175, 87, 459}, {87, 108, 902}, {96, 152, 126}, {6, 63, 358}, {182, 133, 520}, {127, 243, 492}, {96, 69, 7}, {131, 7, 976}, {247, 267, 972}, {224, 152, 383}, {217, 250, 267}, {270, 247, 312}, {157, 3, 613}, {113, 201, 987}, {155, 251, 955}, {21, 144, 456}, {232, 237, 677}, {52, 32, 104}, {20, 137, 68}, {115, 217, 21}, {37, 34, 521}, {70, 160, 533}, {149, 270, 241}, {91, 160, 868}, {184, 149, 3}, {120, 236, 106}, {255, 1, 690}, {254, 206, 84}, {77, 14, 305}, {31, 26, 295}, {105, 61, 832}, {60, 56, 986}, {97, 25, 564}, {2, 29, 349}, {1, 81, 679}, {236, 49, 955}, {161, 14, 497}, {273, 109, 290}, {226, 59, 373}, {206, 194, 197}, {235, 220, 132}, {179, 81, 255}, {32, 234, 690}, {245, 237, 302}, {148, 94, 41}, {265, 3, 586}, {63, 107, 151}, {247, 256, 532}, {234, 56, 278}, {111, 65, 881}, {271, 68, 669}, {10, 127, 8}, {27, 103, 203}, {235, 64, 14}, {191, 177, 993}, {270, 137, 216}},
                new int[][]{{175, 137, 386}, {69, 79, 443}, {199, 76, 975}, {119, 136, 602}, {73, 153, 234}, {219, 154, 889}, {50, 26, 811}, {69, 6, 93}, {156, 22, 734}, {17, 131, 187}, {197, 261, 600}, {105, 74, 286}, {76, 193, 333}, {168, 50, 754}, {64, 30, 721}, {151, 182, 457}, {79, 25, 652}, {124, 271, 486}, {133, 76, 706}, {85, 78, 91}, {33, 98, 268}, {271, 166, 188}, {8, 38, 313}, {245, 109, 488}, {213, 51, 852}, {116, 275, 559}, {214, 244, 568}, {131, 26, 324}, {215, 218, 840}, {11, 16, 701}, {105, 119, 145}, {230, 125, 2}, {66, 222, 773}, {224, 78, 438}, {52, 79, 687}, {165, 117, 476}, {189, 67, 710}, {49, 268, 505}, {126, 237, 646}, {37, 273, 65}, {119, 146, 29}, {268, 129, 42}, {156, 120, 116}, {253, 160, 550}, {172, 258, 532}, {133, 149, 535}, {134, 70, 627}, {47, 11, 559}, {48, 260, 935}, {124, 151, 661}, {151, 9, 3}, {25, 74, 644}, {31, 255, 999}, {46, 100, 303}, {5, 209, 205}, {234, 228, 774}, {4, 155, 532}, {54, 10, 910}, {34, 262, 46}, {131, 210, 100}, {130, 218, 138}, {28, 12, 781}, {223, 164, 996}, {218, 35, 19}, {41, 54, 491}, {136, 246, 136}, {125, 271, 583}, {19, 149, 192}, {34, 36, 809}, {188, 78, 46}, {90, 149, 545}, {163, 31, 133}, {264, 60, 462}, {116, 42, 671}, {93, 176, 937}, {270, 65, 222}, {23, 74, 99}, {43, 37, 90}, {103, 38, 728}, {223, 264, 218}, {14, 223, 78}, {181, 211, 911}, {228, 78, 53}, {72, 136, 72}, {258, 253, 377}, {13, 43, 450}, {79, 224, 807}, {171, 204, 820}, {71, 102, 959}, {82, 33, 378}, {78, 38, 470}, {145, 218, 997}, {72, 58, 276}, {80, 254, 25}, {251, 66, 560}, {65, 226, 359}, {267, 236, 189}, {246, 255, 879}, {66, 275, 626}, {202, 42, 860}, {51, 91, 174}, {6, 103, 727}, {15, 216, 870}, {152, 50, 337}, {120, 129, 27}, {153, 219, 687}, {140, 162, 597}, {33, 227, 181}, {249, 95, 413}, {143, 223, 810}, {255, 197, 835}, {117, 107, 914}, {100, 137, 419}, {142, 121, 150}, {212, 262, 850}, {5, 115, 241}, {82, 85, 958}, {134, 120, 167}, {178, 181, 894}, {41, 23, 81}, {261, 28, 682}, {110, 262, 105}, {176, 191, 372}, {207, 223, 12}, {274, 159, 152}, {38, 35, 886}, {255, 248, 474}, {233, 228, 285}, {44, 181, 938}, {271, 7, 997}, {213, 189, 961}, {0, 212, 332}, {207, 112, 533}, {240, 59, 263}, {255, 189, 406}, {164, 136, 587}, {223, 73, 648}, {218, 201, 637}, {211, 98, 477}, {92, 216, 469}, {106, 127, 741}, {93, 154, 505}, {23, 219, 497}, {15, 206, 84}, {208, 62, 630}, {257, 139, 343}, {258, 155, 599}, {235, 214, 243}, {3, 215, 558}, {77, 18, 79}, {158, 127, 681}, {252, 174, 96}, {230, 167, 961}, {189, 114, 186}, {66, 168, 913}, {133, 237, 985}, {206, 33, 124}, {70, 62, 939}, {107, 133, 629}, {75, 256, 368}, {271, 159, 753}, {97, 99, 986}, {43, 191, 948}, {123, 245, 877}, {8, 142, 877}, {229, 164, 952}, {272, 96, 629}, {201, 262, 246}, {257, 70, 12}, {87, 136, 580}, {19, 33, 840}, {130, 22, 509}, {102, 115, 131}, {240, 24, 724}, {30, 248, 560}, {116, 100, 587}, {27, 241, 577}, {116, 225, 333}, {2, 47, 775}, {241, 28, 876}, {170, 189, 627}, {48, 53, 763}, {105, 88, 739}, {276, 33, 166}, {151, 160, 402}, {126, 20, 369}, {269, 112, 894}, {222, 262, 391}, {11, 93, 314}, {53, 132, 328}, {77, 170, 880}, {16, 123, 83}, {12, 275, 520}, {202, 224, 204}, {40, 238, 829}, {117, 21, 583}, {125, 35, 515}, {93, 141, 133}, {163, 161, 527}, {164, 248, 775}, {173, 61, 304}, {66, 40, 740}, {250, 198, 719}, {99, 96, 297}, {5, 62, 588}, {73, 247, 320}, {187, 50, 920}, {274, 166, 918}, {182, 174, 264}, {253, 236, 867}, {107, 169, 224}, {200, 92, 922}, {248, 221, 529}, {231, 247, 613}, {21, 265, 118}, {39, 113, 535}, {245, 15, 423}, {214, 111, 7}, {99, 0, 228}, {189, 18, 904}, {249, 211, 652}, {113, 242, 344}, {248, 153, 220}, {66, 118, 813}, {154, 132, 607}, {191, 276, 758}, {166, 194, 282}, {72, 251, 241}, {225, 117, 677}, {57, 5, 491}, {14, 268, 114}, {151, 93, 608}, {89, 64, 828}, {113, 250, 749}, {32, 77, 811}, {215, 4, 917}, {195, 248, 216}, {170, 211, 356}, {193, 276, 479}, {205, 168, 36}, {273, 53, 697}, {118, 272, 123}, {192, 122, 678}, {116, 48, 576}, {109, 231, 451}, {261, 85, 402}, {28, 82, 758}, {51, 91, 507}, {200, 11, 556}, {13, 12, 475}, {86, 233, 469}, {230, 22, 714}, {73, 84, 412}, {186, 254, 524}, {13, 100, 139}, {68, 77, 624}, {19, 220, 416}, {163, 137, 513}, {16, 210, 379}, {38, 153, 978}, {150, 245, 552}, {111, 208, 837}, {265, 208, 113}, {156, 188, 268}, {46, 107, 206}, {32, 161, 234}, {77, 83, 492}, {168, 269, 553}, {186, 160, 236}, {177, 134, 20}, {172, 194, 435}, {259, 5, 72}, {94, 7, 951}, {77, 107, 289}, {195, 251, 223}, {61, 22, 521}, {136, 30, 43}, {81, 276, 171}, {207, 79, 738}, {22, 229, 897}, {46, 87, 518}, {210, 22, 836}, {86, 152, 796}, {88, 74, 454}, {17, 53, 782}, {227, 179, 787}, {275, 49, 335}, {79, 90, 472}, {142, 141, 759}, {261, 231, 879}, {28, 52, 427}, {27, 154, 766}, {99, 195, 592}, {198, 128, 648}, {212, 163, 518}, {245, 125, 960}, {67, 272, 110}, {241, 145, 240}, {31, 149, 992}, {250, 173, 758}, {128, 135, 257}, {1, 0, 392}, {178, 250, 656}, {214, 51, 573}, {226, 64, 100}, {216, 67, 636}, {102, 130, 582}, {14, 181, 991}, {103, 268, 105}, {220, 202, 179}, {21, 245, 824}, {210, 87, 137}, {178, 203, 301}, {138, 192, 782}, {179, 214, 403}, {12, 77, 39}, {108, 22, 877}, {85, 145, 519}, {113, 5, 586}, {80, 92, 347}, {61, 234, 33}, {142, 58, 219}, {223, 245, 807}, {117, 266, 996}, {45, 91, 73}, {264, 112, 615}, {155, 226, 710}, {246, 169, 259}, {160, 178, 521}, {238, 263, 55}, {144, 134, 329}, {56, 217, 355}, {204, 114, 971}, {171, 23, 799}, {132, 207, 876}, {187, 36, 201}, {116, 247, 304}, {171, 218, 884}, {239, 30, 29}, {28, 264, 302}, {93, 6, 417}, {60, 128, 70}, {3, 87, 144}, {270, 132, 582}, {155, 64, 88}, {209, 109, 722}, {73, 242, 427}, {215, 262, 269}}
        ));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1697 TLE
    public boolean[] distanceLimitedPathsExist(int n, int[][] edgeList, int[][] queries) {
        boolean[] result = new boolean[queries.length];
        List<Map<Integer, Integer>> matrix = new ArrayList<>(n);
        // [key,value] : key: 下一个点, value: 距离
        for (int i = 0; i < n; i++) matrix.add(new HashMap<>());
        for (int[] e : edgeList) {
            if (!matrix.get(e[0]).containsKey(e[1]) || matrix.get(e[0]).get(e[1]) > e[2]) {
                matrix.get(e[0]).put(e[1], e[2]);
            }
            if (!matrix.get(e[1]).containsKey(e[0]) || matrix.get(e[1]).get(e[0]) > e[2]) {
                matrix.get(e[1]).put(e[0], e[2]);
            }
        }
        List<TreeSet<Pair<Integer, Integer>>> mtx = new ArrayList<>();
        for (int i = 0; i < n; i++) mtx.add(new TreeSet<>(Comparator.comparingInt(o -> o.getValue())));
        for (int i = 0; i < n; i++) {
            for (Map.Entry<Integer, Integer> e : matrix.get(i).entrySet()) {
                mtx.get(i).add(new Pair<>(e.getKey(), e.getValue()));
            }
        }
        DisjointSetUnion<Integer> dsu = new DisjointSetUnion<>();
        for (int i = 0; i < n; i++) dsu.add(i);

        Map<int[], Integer> queryIdxMap = new HashMap<>();
        for (int i = 0; i < queries.length; i++) {
            queryIdxMap.put(queries[i], i);
        }
        Arrays.sort(queries, Comparator.comparingInt(o -> o[2]));
        for (int i = 0; i < queries.length; i++) {
            int[] q = queries[i];
            int start = q[0], end = q[1], limit = q[2];
            for (int j = 0; j < n; j++) {
                NavigableSet<Pair<Integer, Integer>> pairs = mtx.get(j).subSet(new Pair<>(0, 0), true, new Pair<>(0, limit), false);
                for (Pair<Integer, Integer> p : pairs) {
                    dsu.merge(p.getKey(), j);
                }
                if (dsu.isConnected(start, end)) {
                    result[queryIdxMap.get(q)] = true;
                    break;
                }
            }
        }
        return result;
    }

    // LC1769
    public int[] minOperations(String boxes) {
        int n = boxes.length();
        int leftOper = 0, leftOneCount = 0, rightOper = 0, rightOneCount = 0;
        int[] result = new int[n];
        for (int i = 0; i < n; i++) {
            if (boxes.charAt(i) == '1') {
                rightOneCount++;
                rightOper += i;
            }
        }
        result[0] = rightOper;
        for (int i = 1; i < n; i++) {
            if (boxes.charAt(i - 1) == '1') {
                leftOneCount++;
                rightOneCount--;
            }
            leftOper += leftOneCount; // 每往左移一步, 左边的"1"到该点的路程各自加1, 总共加左边1的个数
            rightOper -= rightOneCount; // 如果这个位置本身是1, 通过这里的减, 其实自身到自身的路程已经为0了
            result[i] = leftOper + rightOper;
        }
        return result;
    }

    // LCP33 **
    public int storeWater(int[] bucket, int[] vat) {
        int n = bucket.length, maxVat = Integer.MIN_VALUE;
        for (int i = 0; i < n; i++) maxVat = Math.max(maxVat, vat[i]);
        if (maxVat == 0) return 0;
        int minOper = Integer.MAX_VALUE;
        for (int i = 1; i <= maxVat; i++) {
            minOper = Math.min(minOper, getOper(bucket, vat, i));
        }
        return minOper;
    }

    private int getOper(int[] bucket, int[] vat, int stores) { // stores: 倒水次数
        int n = bucket.length, oper = 0;
        for (int i = 0; i < n; i++) {
            int buck = vat[i] / stores + (vat[i] % stores == 0 ? 0 : 1); // 上取整, 即倒水之前要加到多少水
            oper += (buck > bucket[i] ? buck - bucket[i] : 0); // 如果已经超过最小值就不用再加水了
        }
        return stores + oper;
    }

    private boolean check(long[] cur, int[] vat) {
        for (int i = 0; i < cur.length; i++) {
            if (cur[i] < vat[i]) return false;
        }
        return true;
    }

    // LC358 ** Heap
    public String rearrangeString(String s, int k) {
        if (k == 0) return s;
        PriorityQueue<Pair<Character, Integer>> pq = new PriorityQueue<>(Comparator.comparingInt(o -> -o.getValue()));
        Map<Character, Integer> freq = new HashMap<>(26);
        Deque<Pair<Character, Integer>> q = new LinkedList<>();
        StringBuilder sb = new StringBuilder();
        for (char c : s.toCharArray()) freq.put(c, freq.getOrDefault(c, 0) + 1);
        for (Map.Entry<Character, Integer> e : freq.entrySet()) pq.offer(new Pair<>(e.getKey(), e.getValue()));
        while (!pq.isEmpty()) {
            Pair<Character, Integer> p = pq.poll();
            sb.append(p.getKey());
            q.offer(new Pair<>(p.getKey(), p.getValue() - 1));
            if (q.size() == k) {
                if (q.peek().getValue() > 0) {
                    pq.offer(q.peek());
                }
                q.poll();
            }
        }
        return sb.length() < s.length() ? "" : sb.toString();
    }


    // LC453
    public int minMoves(int[] nums) {
        // 每次n-1个数增加1, 问增加多少次相等
        // 等同于每次挑一个数减少1, 问总共减少多少次相等
        int min = Integer.MAX_VALUE;
        long sum = 0;
        for (int i : nums) {
            min = Math.min(min, i);
            sum += i;
        }
        return (int) (sum - (min + 0l) * (nums.length + 0l));
    }

    // LC1093
    public double[] sampleStats(int[] count) {
        // min max avg middle popular
        double[] result = new double[5];
        int total = 0;
        for (int i : count) total += i;
        if (total % 2 == 0) {
            // 找 half, half+1
            int first = -1, second = -1;
            int half = total / 2;
            int accu = 0;
            for (int i = 0; i < 256; i++) {
                accu += count[i];
                if (first == -1 && accu >= half) {
                    first = i;
                }
                if (first != -1 && accu >= half + 1) {
                    second = i;
                    break;
                }
            }
            result[3] = (first + second + 0d) / 2d;
        } else {
            // 找half+1
            int first = -1;
            int half = total / 2;
            int accu = 0;
            for (int i = 0; i < 256; i++) {
                accu += count[i];
                if (first == -1 && accu >= half + 1) {
                    first = i;
                    break;
                }
            }
            result[3] = first;
        }
        for (int i = 0; i < 256; i++) {
            if (count[i] != 0) {
                result[0] = i;
                break;
            }
        }
        for (int i = 255; i >= 0; i--) {
            if (count[i] != 0) {
                result[1] = i;
                break;
            }
        }
        long sum = 0;
        for (int i = 0; i < 256; i++) {
            sum += (count[i] + 0l) * (i + 0l);
        }
        result[2] = (sum + 0d) / (total + 0d);
        int popCount = 0, popNum = -1;
        for (int i = 0; i < 256; i++) {
            if (count[i] > popCount) {
                popNum = i;
                popCount = count[i];
            }
        }
        result[4] = popNum;
        return result;
    }

    // LC1058
    public String minimizeError(String[] prices, int target) {
        double max = 0d, min = 0d, sum = 0d;
        List<Double> priceList = new ArrayList<>(prices.length);
        for (String p : prices) {
            double d = Double.parseDouble(p);
            if (p.endsWith("000")) { // 整数会对shouldFloorCount造成影响, 导致shouldFloorCount少1, 最终的和会变大
                target -= (int) d;
                continue;
            }
            priceList.add(d);
            max += Math.ceil(d);
            min += Math.floor(d);
            sum += d;
        }
        if (max < (target + 0d)) return "-1";
        if (min > (target + 0d)) return "-1";
        int shouldFloorCount = (int) (max - target);
        if (shouldFloorCount == 0) {
            return String.format("%.3f", max - sum);
        }
        if (shouldFloorCount == prices.length) {
            return String.format("%.3f", sum - min);
        }
        PriorityQueue<Double> floorPq = new PriorityQueue<>(Comparator.comparingDouble(d -> -(d - Math.floor(d))));
        List<Double> shouldCeilList = new ArrayList<>(priceList.size() - shouldFloorCount);
        for (double d : priceList) {
            if (floorPq.size() < shouldFloorCount) {
                floorPq.offer(d);
            } else {
                double deltaOut = d - Math.floor(d);
                double peek = floorPq.peek();
                double deltaIn = peek - Math.floor(peek);
                if (deltaOut < deltaIn) {
                    floorPq.poll();
                    floorPq.offer(d);
                    shouldCeilList.add(peek);
                } else {
                    shouldCeilList.add(d);
                }
            }
        }
        double totalDelta = 0d;
        while (!floorPq.isEmpty()) {
            double p = floorPq.poll();
            totalDelta += p - Math.floor(p);
        }
        for (double d : shouldCeilList) {
            totalDelta += Math.ceil(d) - d;
        }
        return String.format("%.3f", totalDelta);
    }


    // LC1508 前缀和 + 暴力
    public int rangeSum(int[] nums, int n, int left, int right) {
        int[] prefix = new int[n + 1];
        for (int i = 0; i < n; i++) prefix[i + 1] = prefix[i] + nums[i];
        List<Integer> accu = new ArrayList<>((n + 1) * n / 2);
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                accu.add(prefix[j + 1] - prefix[i]);
            }
        }
        Collections.sort(accu);
        long sum = 0, mod = 1000000007;
        for (int i = left - 1; i < right; i++) {
            sum += accu.get(i);
            sum %= mod;
        }
        return (int) (sum % mod);
    }

    // LC1078
    public String[] findOcurrences(String text, String first, String second) {
        List<String> result = new ArrayList<>();
        String[] words = text.split(" ");
        String prev = '\0' + "";
        for (int i = 0; i < words.length; i++) {
            String cur = words[i];
            if (prev.equals(first) && cur.equals(second)) {
                if (i + 1 < words.length) {
                    result.add(words[i + 1]);
                }
            }
            prev = cur;
        }
        return result.toArray(new String[result.size()]);
    }

    // LC1815 Try Simulate Anneal 模拟退火
    class Lc1815 {
        List<Integer> toSatisfy = new ArrayList<>();
        int max = 0;
        int n;
        int batchSize;

        public int maxHappyGroups(int batchSize, int[] groups) {
            if (batchSize == 1) return groups.length;
            this.batchSize = batchSize;
            int modZeroGroupCount = 0;
            for (int i : groups) {
                if (i % batchSize == 0) modZeroGroupCount++;
                else toSatisfy.add(i);
            }
            if (toSatisfy.size() < 2) return modZeroGroupCount + toSatisfy.size();
            n = toSatisfy.size();
            // 多次使用模拟退火
            for (int i = 0; i < 32; i++) {
                simulateAnneal();
            }
            return modZeroGroupCount + max;
        }

        private void simulateAnneal() {
            Collections.shuffle(toSatisfy); // 随机化
            for (double t = 1e6; t >= 1e-5; t *= 0.97d) {
                int i = (int) (n * Math.random()), j = (int) (n * Math.random());
                int prevScore = evaluate();
                // 交换
                int tmp = toSatisfy.get(i);
                toSatisfy.set(i, toSatisfy.get(j));
                toSatisfy.set(j, tmp);

                int nextScore = evaluate();
                int delta = nextScore - prevScore;
                if (delta < 0 && Math.pow(Math.E, (delta / t)) <= (double) (Math.random())) {
                    tmp = toSatisfy.get(i);
                    toSatisfy.set(i, toSatisfy.get(j));
                    toSatisfy.set(j, tmp);
                }
            }
        }

        private int evaluate() {
            // 评价函数 直接返回在当前排列下有多少组可以好评
            int result = 0, sum = 0;
            for (int i : toSatisfy) {
                sum += i;
                if (sum % batchSize == 0) {
                    result++; // 能收到这一组的好评
                    sum = 0;
                }
            }
            if (sum > 0) result++; // 总是能收到第一组的好评
            max = Math.max(max, result);
            return result;
        }

    }


    // LC286
    public void wallsAndGates(int[][] rooms) {
        int m = rooms.length, n = rooms[0].length;
        final int INF = Integer.MAX_VALUE;
        int[][] direction = new int[][]{{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        int layer = -1;
        boolean[][] visited = new boolean[m][n];
        Deque<int[]> q = new LinkedList<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (rooms[i][j] == 0) {
                    q.offer(new int[]{i, j});
                }
            }
        }
        while (!q.isEmpty()) {
            int qs = q.size();
            layer++;
            for (int i = 0; i < qs; i++) {
                int[] p = q.poll();
                if (visited[p[0]][p[1]]) continue;
                visited[p[0]][p[1]] = true;
                if (rooms[p[0]][p[1]] == INF) rooms[p[0]][p[1]] = layer;
                for (int[] d : direction) {
                    int nr = p[0] + d[0], nc = p[1] + d[1];
                    if (nr >= 0 && nr < m && nc >= 0 && nc < n && !visited[nr][nc] && rooms[nr][nc] != -1) {
                        q.offer(new int[]{nr, nc});
                    }
                }
            }
        }
        return;
    }

    // LC1976 **
    final long lc1976Inf = Long.MAX_VALUE / 2;
    Integer[] lc1976Memo;

    public int countPaths(int n, int[][] roads) {
        // 最短路的数量
        lc1976Memo = new Integer[n];
        long[][] matrix = new long[n][n];
        long[][] minDist = new long[n][n];
        for (int i = 0; i < n; i++) {
            Arrays.fill(matrix[i], lc1976Inf);
            Arrays.fill(minDist[i], lc1976Inf);
        }
        for (int[] r : roads) {
            matrix[r[0]][r[1]] = r[2];
            matrix[r[1]][r[0]] = r[2];
            minDist[r[0]][r[1]] = r[2];
            minDist[r[1]][r[0]] = r[2];
        }
        for (int i = 0; i < n; i++) {
            minDist[i][i] = 0;
        }
        // Floyd 求出任意两点间的最短距离
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    if (j != k) {
                        minDist[j][k] = minDist[k][j] = Math.min(minDist[j][k], minDist[j][i] + minDist[i][k]);
                    }
                }
            }
        }
        return lc1976Helper(0, minDist, matrix);
    }

    private int lc1976Helper(int cur, long[][] minDist, long[][] matrix) {
        if (cur == minDist.length - 1) return 1; // 已经到达最后一个节点
        if (lc1976Memo[cur] != null) return lc1976Memo[cur];
        int n = minDist.length;
        long result = 0;
        final long mod = 1000000007;
        for (int next = 0; next < n; next++) {
            if (matrix[cur][next] != lc1976Inf && minDist[0][cur] + minDist[next][n - 1] + matrix[cur][next] == minDist[0][n - 1]) {
                result += lc1976Helper(next, minDist, matrix);
                result %= mod;
            }
        }
        return lc1976Memo[cur] = (int) (result % mod);
    }

    // LC1017 **
    public String baseNeg2(int n) {
        StringBuilder sb = new StringBuilder();
        List<Integer> result = toBase(n, -2);
        for (int i : result) sb.append(i);
        return sb.toString();
    }

    public List<Integer> toBase(int num, int base) {
        if (num == 0) return Arrays.asList(0);
        List<Integer> result = new ArrayList<>();
        while (num != 0) {
            int r = ((num % base) + Math.abs(base)) % Math.abs(base);
            result.add(r);
            num -= r;
            num /= base;
        }
        Collections.reverse(result);
        return result;
    }

    // LC1256
    public String encode(int num) {
        // 0 -> ""
        // 2 -> 0
        // 3 -> 1
        // 4 -> 00
        // 5 -> 01
        // 6 -> 10
        // 7 -> 11
        // 8 -> 000
        if (num == 0) return "";
        num++;
        boolean flag = false;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 32; i++) {
            if ((num >> (32 - i - 1) & 1) == 1) {
                if (!flag) {
                    flag = true;
                    continue;
                }
            }
            if (flag) {
                sb.append(num >> (32 - i - 1) & 1);
            }
        }
        return sb.toString();
    }

    // LC1218
    Map<Integer, TreeSet<Integer>> lc1218IdxMap;
    Integer[] lc1218Memo;

    public int longestSubsequence(int[] arr, int difference) {
        int n = arr.length;
        boolean[] visited = new boolean[n];
        lc1218IdxMap = new HashMap<>();
        lc1218Memo = new Integer[n + 1];
        int max = 1;
        for (int i = 0; i < n; i++) {
            lc1218IdxMap.putIfAbsent(arr[i], new TreeSet<>());
            lc1218IdxMap.get(arr[i]).add(i);
        }
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                int result = 1 + lc1218Helper(i, difference, arr, visited);
                max = Math.max(max, result);
            }
        }
        return max;
    }

    private int lc1218Helper(int idx, int difference, int[] arr, boolean[] visited) {
        visited[idx] = true;
        if (lc1218Memo[idx] != null) return lc1218Memo[idx];
        int expected = arr[idx] + difference;
        if (lc1218IdxMap.get(expected) != null) {
            Integer nextIdx = lc1218IdxMap.get(expected).higher(idx);
            if (nextIdx != null) {
                return lc1218Memo[idx] = 1 + lc1218Helper(nextIdx, difference, arr, visited);
            }
        }
        return lc1218Memo[idx] = 0;
    }

    // JZOF II 102 LC494
    public int findTargetSumWays(int[] nums, int target) {
        int sum = 0;
        for (int i : nums) sum += i;
        if (Math.abs(target) > sum) return 0;
        int OFFSET = sum, n = nums.length;
        int[][] dp = new int[2][OFFSET * 2 + 1];
        dp[0][OFFSET] = 1; // 加入0个数, 和为 (0+OFFSET) 的个数为0
        for (int i = 1; i <= n; i++) {
            for (int total = 0; total <= 2 * sum; total++) {
                int result = 0;
                // 背包问题
                if (total - nums[i - 1] >= 0) {
                    result += dp[(i - 1) % 2][total - nums[i - 1]];
                }
                if (total + nums[i - 1] <= 2 * OFFSET) {
                    result += dp[(i - 1) % 2][total + nums[i - 1]];
                }
                dp[i % 2][total] = result;
            }
        }
        return dp[n % 2][OFFSET + target];
    }

    // LC1087
    public String[] expand(String s) {
        List<String> l = braceExpansionII(s);
        return l.toArray(new String[l.size()]);
    }

    // LC1096 ** DFS
    public List<String> braceExpansionII(String expression) {
        Set<String> result = helper(expression);
        List<String> l = new ArrayList<>(result);
        Collections.sort(l);
        return l;
    }

    private Set<String> helper(String chunk) {
        if (chunk.length() == 0) return new HashSet<>();
        Set<String> result = new HashSet<>();
        Set<String> peek = new HashSet<>();
        char[] ca = chunk.toCharArray();
        int i = 0, n = ca.length;
        while (i < n) {
            char c = ca[i];
            if (c == '{') {
                int numParenthesis = 1;
                int start = ++i; // 括号对内的起始下标(不包括括号)
                while (numParenthesis != 0) {
                    if (ca[i] == '{') numParenthesis++;
                    if (ca[i] == '}') numParenthesis--;
                    i++;
                }
                Set<String> next = helper(chunk.substring(start, i - 1));
                peek = merge(peek, next);
                continue;
            } else if (c == ',') {
                result.addAll(peek);
                peek.clear();
                i++;
                continue;
            } else { // 不会遍历到 '{'
                StringBuilder word = new StringBuilder();
                while (i < n && Character.isLetter(ca[i])) {
                    word.append(ca[i]);
                    i++;
                }
                Set<String> tmp = new HashSet<>();
                tmp.add(word.toString());
                peek = merge(peek, tmp);
            }
        }
        if (i == n) result.addAll(peek);
        return result;
    }

    private Set<String> merge(Set<String> prefix, Set<String> suffix) {
        if (suffix.size() == 0) return prefix;
        if (prefix.size() == 0) return suffix;
        Set<String> result = new HashSet<>();
        for (String p : prefix) {
            for (String s : suffix) {
                result.add(p + s);
            }
        }
        return result;
    }

    // ** BFS
    public List<String> braceExpansionIiBfs(String expression) {
        expression = "{" + expression + "}"; // 预防 "a,{b}c"这种情况
        Deque<String> q = new LinkedList<>();
        q.offer(expression);
        Set<String> result = new HashSet<>();
        while (!q.isEmpty()) {
            String p = q.poll();
            if (p.indexOf("{") < 0) {
                result.add(p);
                continue;
            }
            // ** 找最深的括号对
            int idx = 0, left = -1, right = -1;
            while (p.charAt(idx) != '}') {
                if (p.charAt(idx) == '{') left = idx;
                idx++;
            }
            right = idx;
            String prefix = p.substring(0, left);
            String suffix = p.substring(right + 1);
            String[] middle = p.substring(left + 1, right).split(",");

            for (String m : middle) {
                q.offer(prefix + m + suffix);
            }
        }
        List<String> l = new ArrayList<>(result);
        Collections.sort(l);
        return l;
    }

    // LC1807
    public String evaluate(String s, List<List<String>> knowledge) {
        StringBuilder sb = new StringBuilder();
        char[] ca = s.toCharArray();
        Map<String, String> map = new HashMap<>(knowledge.size());
        for (List<String> k : knowledge) {
            map.put(k.get(0), k.get(1));
        }
        int left = -1, right = -1;
        boolean inParenthesis = false;
        for (int i = 0; i < ca.length; i++) {
            char c = ca[i];
            if (c == '(') {
                left = i;
                inParenthesis = true;
            } else if (c == ')') {
                right = i;
                inParenthesis = false;
                String key = s.substring(left + 1, right);
                sb.append(map.getOrDefault(key, "?"));
            } else {
                if (inParenthesis) continue;
                sb.append(c);
            }
        }
        return sb.toString();
    }

    // LC814
    public TreeNode pruneTree(TreeNode root) {
        if (!subtreeHasOne(root)) return null;
        lc814Helper(root);
        return root;
    }

    private void lc814Helper(TreeNode root) {
        if (root == null) return;
        if (!subtreeHasOne(root.left)) {
            root.left = null;
        } else {
            lc814Helper(root.left);
        }
        if (!subtreeHasOne(root.right)) {
            root.right = null;
        } else {
            lc814Helper(root.right);
        }
    }

    private boolean subtreeHasOne(TreeNode root) {
        if (root == null) return false;
        if (root.val == 1) return true;
        return subtreeHasOne(root.left) || subtreeHasOne(root.right);
    }

    // LC306
    public boolean isAdditiveNumber(String num) {
        int n = num.length();
        // 选前两个数
        for (int i = 1; i <= n / 2; i++) {
            long first = Long.parseLong(num.substring(0, i));
            if (String.valueOf(first).length() != i) continue;
            for (int j = i + 1; j < n; j++) {
                long second = Long.parseLong(num.substring(i, j));
                if (String.valueOf(second).length() != j - i) continue;
                if (judge(first, second, j, num)) return true;
            }
        }
        return false;
    }

    private boolean judge(long first, long second, int idx, String num) {
        if (idx == num.length()) return true;
        long sum = first + second;
        if (num.indexOf(String.valueOf(sum), idx) != idx) return false;
        return judge(second, sum, idx + String.valueOf(sum).length(), num);
    }


    // LC311 矩阵乘法
    public int[][] multiply(int[][] mat1, int[][] mat2) {
        // axb mult bxc = axc
        int a = mat1.length, b = mat1[0].length, c = mat2[0].length;
        int[][] result = new int[a][c];
        for (int i = 0; i < a; i++) {
            for (int k = 0; k < b; k++) {
                if (mat1[i][k] == 0) continue;
                for (int j = 0; j < c; j++) {
                    result[i][j] += mat1[i][k] * mat2[k][j];
                }
            }
        }
        return result;
    }

    // LC259 ** Solution O(n^2)
    public int threeSumSmaller(int[] nums, int target) {
        int n = nums.length;
        Arrays.sort(nums);
        int result = 0;
        for (int i = 0; i < n - 2; i++) {
            result += twoSumSmaller(nums, i + 1, target - nums[i]);
        }
        return result;
    }

    private int twoSumSmaller(int[] nums, int startIdx, int target) {
        int result = 0;
        int left = startIdx, right = nums.length - 1;
        while (left < right) {
            if (nums[left] + nums[right] < target) {
                result += right - left;
                left++;
            } else {
                right--;
            }
        }
        return result;
    }

    // LC1243
    public List<Integer> transformArray(int[] arr) {
        int n = arr.length;
        List<Integer> prev = Arrays.stream(arr).boxed().collect(Collectors.toList());
        List<Integer> cur = new ArrayList<>();
        for (int i = 0; i < n; i++) cur.add(-1);
        while (true) {
            cur = helper(prev);
            if (cur.equals(prev)) return cur;
            prev = cur;
        }
    }

    private List<Integer> helper(List<Integer> prev) {
        int n = prev.size();
        List<Integer> cur = new ArrayList<>();
        cur.add(prev.get(0));
        for (int i = 1; i < n - 1; i++) {
            // 假如一个元素小于它的左右邻居，那么该元素自增 1。
            // 假如一个元素大于它的左右邻居，那么该元素自减 1。
            if (prev.get(i) < prev.get(i - 1) && prev.get(i) < prev.get(i + 1)) {
                cur.add(prev.get(i) + 1);
            } else if (prev.get(i) > prev.get(i - 1) && prev.get(i) > prev.get(i + 1)) {
                cur.add(prev.get(i) - 1);
            } else {
                cur.add(prev.get(i));
            }
        }
        cur.add(prev.get(n - 1));
        return cur;
    }


    // Interview 17.09 LC264 UglyNumber 丑数
    public int getKthMagicNumber(int k) {
        // Prime Factor 3,5,7
        long[] factor = {3, 5, 7};
        PriorityQueue<Long> pq = new PriorityQueue<>();
        Set<Long> set = new HashSet<>();
        pq.offer(1l);
        set.add(1l);
        long result = -1;
        for (int i = 0; i < k; i++) {
            long p = pq.poll();
            result = p;
            for (long f : factor) {
                if (set.add(f * p)) {
                    pq.offer(f * p);
                }
            }
        }
        return (int) result;
    }

    // LC365
    public boolean canMeasureWater(int jug1Capacity, int jug2Capacity, int targetCapacity) {
        Deque<int[]> q = new LinkedList<>();
        Set<Pair<Integer, Integer>> visited = new HashSet<>();
        q.offer(new int[]{0, 0});
        q.offer(new int[]{jug1Capacity, jug2Capacity});
        while (!q.isEmpty()) {
            int[] p = q.poll();
            Pair<Integer, Integer> pair = new Pair<>(p[0], p[1]);
            if (visited.contains(pair)) continue;
            visited.add(pair);
            if (p[0] == targetCapacity || p[1] == targetCapacity) return true;
            if (p[0] + p[1] == targetCapacity) return true;
            // 倒满一侧
            pair = new Pair<>(jug1Capacity, p[1]);
            if (!visited.contains(pair)) {
                q.offer(new int[]{jug1Capacity, p[1]});
            }
            pair = new Pair<>(p[0], jug2Capacity);
            if (!visited.contains(pair)) {
                q.offer(new int[]{p[0], jug2Capacity});
            }
            // 倒掉一侧
            pair = new Pair<>(0, p[1]);
            if (!visited.contains(pair)) {
                q.offer(new int[]{0, p[1]});
            }
            pair = new Pair<>(p[0], 0);
            if (!visited.contains(pair)) {
                q.offer(new int[]{p[0], 0});
            }
            // 一侧倒向另一侧
            if (p[0] < jug1Capacity) {
                int jug1Empty = jug1Capacity - p[0];
                int jug2ToJug1 = Math.min(p[1], jug1Empty);
                pair = new Pair<>(p[0] + jug2ToJug1, p[1] - jug2ToJug1);
                if (!visited.contains(pair)) {
                    q.offer(new int[]{p[0] + jug2ToJug1, p[1] - jug2ToJug1});
                }
            }
            if (p[1] < jug2Capacity) {
                int jug2Empty = jug2Capacity - p[1];
                int jug1ToJug2 = Math.min(p[0], jug2Empty);
                pair = new Pair<>(p[0] - jug1ToJug2, p[1] + jug1ToJug2);
                if (!visited.contains(pair)) {
                    q.offer(new int[]{p[0] - jug1ToJug2, p[1] + jug1ToJug2});
                }
            }
        }
        return false;
    }

    // LC439 ** Great Solution
    public String parseTernary(String expression) {
        int len = expression.length();
        int level = 0;
        for (int i = 1; i < len; i++) {
            if (expression.charAt(i) == '?') level++;
            if (expression.charAt(i) == ':') level--;
            if (level == 0) {
                return expression.charAt(0) == 'T' ?
                        parseTernary(expression.substring(2, i)) : parseTernary(expression.substring(i + 1));
            }
        }
        return expression;
    }

    // LC385
    public NestedInteger deserialize(String s) {
        NestedInteger root = new NestedInteger();
        if (s.charAt(0) != '[') {
            root.setInteger(Integer.parseInt(s));
            return root;
        }
        Deque<NestedInteger> stack = new LinkedList<>();
        StringBuilder sb = new StringBuilder();
        char[] ca = s.toCharArray();
        for (int i = 0; i < ca.length; i++) {
            char c = ca[i];
            if (c == '[') {
                NestedInteger next = new NestedInteger();
                stack.push(next);
            } else if (c == ']') {
                NestedInteger pop = stack.pop();
                if (sb.length() != 0) {
                    pop.add(new NestedInteger(Integer.parseInt(sb.toString())));
                    sb = new StringBuilder();
                }
                if (!stack.isEmpty()) {
                    stack.peek().add(pop);
                    continue;
                } else {
                    return pop;
                }
            } else if (c == ',') {
                NestedInteger peek = stack.peek();
                if (sb.length() != 0) {
                    peek.add(new NestedInteger(Integer.parseInt(sb.toString())));
                    sb = new StringBuilder();
                }
                continue;
            } else {
                sb.append(c);
            }
        }
        return null;
    }
}

// LC385
class NestedInteger {
    // Constructor initializes an empty nested list.
    public NestedInteger() {

    }

    // Constructor initializes a single integer.
    public NestedInteger(int value) {

    }

    // @return true if this NestedInteger holds a single integer, rather than a nested list.
    public boolean isInteger() {
        return false;
    }

    // @return the single integer that this NestedInteger holds, if it holds a single integer
    // Return null if this NestedInteger holds a nested list
    public Integer getInteger() {
        return -1;
    }

    // Set this NestedInteger to hold a single integer.
    public void setInteger(int value) {
        ;
    }

    // Set this NestedInteger to hold a nested list and adds a nested integer to it.
    public void add(NestedInteger ni) {
        ;
    }

    // @return the nested list that this NestedInteger holds, if it holds a nested list
    // Return empty list if this NestedInteger holds a single integer
    public List<NestedInteger> getList() {
        return null;
    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int x) {
        val = x;
    }
}


class Trie {
    TrieNode root;

    public Trie() {
        root = new TrieNode();
    }


    public boolean addWord(String word) {
        if (search(word)) return false;
        TrieNode cur = root;
        for (char c : word.toCharArray()) {
            if (!cur.children.containsKey(c)) {
                cur.children.put(c, new TrieNode());
            }
            cur = cur.children.get(c);
            cur.path++;
        }
        cur.end++;
        return true;
    }

    public boolean remove(String word) {
        if (!search(word)) return false;
        TrieNode cur = root;
        for (char c : word.toCharArray()) {
            if (cur.children.get(c).path-- == 1) {
                cur.children.remove(c);
                return true;
            }
            cur = cur.children.get(c);
        }
        cur.end--;
        return true;
    }

    public boolean search(String word) {
        TrieNode target = getNode(word);
        return target != null && target.end > 0;
    }

    public boolean beginWith(String prefix) {
        return getNode(prefix) != null;
    }

    private TrieNode getNode(String prefix) {
        TrieNode cur = root;
        for (char c : prefix.toCharArray()) {
            if (!cur.children.containsKey(c)) return null;
            cur = cur.children.get(c);
        }
        return cur;
    }

}


class TrieNode {
    Map<Character, TrieNode> children = new HashMap<>();
    int end = 0;
    int path = 0;
}


// LC211
class WordDictionary {
    Trie trie;

    public WordDictionary() {
        trie = new Trie();
    }

    public void addWord(String word) {
        trie.addWord(word);
    }

    public boolean search(String word) {
        return searchHelper("", word);
    }

    private boolean searchHelper(String prefix, String suffix) {
        if (suffix.equals("")) return trie.search(prefix);
        StringBuilder sb = new StringBuilder(prefix);
        for (int i = 0; i < suffix.length(); i++) {
            if (suffix.charAt(i) != '.') {
                sb.append(suffix.charAt(i));
            } else {
                for (int j = 0; j < 26; j++) {
                    sb.append((char) ('a' + j));
                    if (!trie.beginWith(sb.toString())) {
                        sb.deleteCharAt(sb.length() - 1);
                        continue;
                    }
                    if (searchHelper(sb.toString(), suffix.substring(i + 1))) return true;
                    sb.deleteCharAt(sb.length() - 1);
                }
                // 一旦所有'.'的可能性都被尝试, 且无一匹配, 即可返回false
                return false;
            }
        }
        // 如果全程没有'.', 返回Trie中是否有这个word
        return trie.search(sb.toString());
    }
}

// LC126
class Lc126 {
    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        List<List<String>> result = new ArrayList<>();
        Node start = new Node(null, beginWord);
        Set<String> visited = new HashSet<>(wordList.size());
        Map<String, Set<String>> edge = new HashMap<>();
        for (String w : wordList) edge.put(w, new HashSet<>());
        edge.put(beginWord, new HashSet<>());
        for (int i = 0; i < wordList.size(); i++) {
            for (int j = i + 1; j < wordList.size(); j++) {
                String iw = wordList.get(i), jw = wordList.get(j);
                if (oneLetterDiff(iw.toCharArray(), jw.toCharArray())) {
                    edge.get(iw).add(jw);
                    edge.get(jw).add(iw);
                }
            }
            if (oneLetterDiff(wordList.get(i).toCharArray(), beginWord.toCharArray())) {
                edge.get(beginWord).add(wordList.get(i));
            }
        }

        Deque<Node> q = new LinkedList<>();
        List<Node> resultNodeList = new ArrayList<>();
        int layer = -1, finalLayer = -1;
        q.offer(start);
        while (!q.isEmpty()) {
            int qs = q.size();
            Set<String> thisLayerVisited = new HashSet<>();
            layer++;
            if (finalLayer != -1 && layer > finalLayer) break;
            for (int i = 0; i < qs; i++) {
                Node p = q.poll();
                if (p.word.equals(endWord)) {
                    finalLayer = layer;
                    resultNodeList.add(p);
                    continue;
                }
                if (visited.contains(p.word)) continue;
                thisLayerVisited.add(p.word);
                if (finalLayer == -1) {
                    for (String next : edge.get(p.word)) {
                        if (!visited.contains(next)) {
                            q.offer(new Node(p, next));
                        }
                    }
                }
            }
            // 注意求所有情况的时候不能根据在同一层已经访问过的边剪枝, 应该等该层全部访问后, 再将已访问节点标记
            visited.addAll(thisLayerVisited);
        }
        for (Node resultNode : resultNodeList) {
            List<String> r = new LinkedList<>();
            while (resultNode != null) {
                r.add(0, resultNode.word);
                resultNode = resultNode.prev;
            }
            result.add(r);
        }
        return result;
    }

    private boolean oneLetterDiff(char[] a, char[] b) {
        int ctr = 0;
        for (int i = 0; i < a.length; i++) {
            if (a[i] != b[i]) ctr++;
            if (ctr == 2) return false;
        }
        return true;
    }

    class Node {
        Node prev;
        String word;

        public Node(Node prev, String word) {
            this.prev = prev;
            this.word = word;
        }
    }
}

class DisjointSetUnion<T> {

    Map<T, T> father;
    Map<T, Integer> rank;

    public DisjointSetUnion() {
        father = new HashMap<>();
        rank = new HashMap<>();
    }

    public void add(T i) {
        if (!father.containsKey(i)) {
            // 置初始父亲为自身
            // 之后判断连通分量个数时候, 遍历father, 找value==key的
            father.put(i, i);
        }
        if (!rank.containsKey(i)) {
            rank.put(i, 1);
        }
    }

    // 找父亲, 路径压缩
    public T find(T i) {
        //先找到根 再压缩
        T root = i;
        while (father.get(root) != root) {
            root = father.get(root);
        }
        // 找到根, 开始对一路上的子节点进行路径压缩
        while (father.get(i) != root) {
            T origFather = father.get(i);
            father.put(i, root);
            // 更新秩, 按照节点数
            rank.put(root, rank.get(root) + 1);
            i = origFather;
        }
        return root;
    }

    public boolean merge(T i, T j) {
        T iFather = find(i);
        T jFather = find(j);
        if (iFather == jFather) return false;
        // 按秩合并
        if (rank.get(iFather) >= rank.get(jFather)) {
            father.put(jFather, iFather);
            rank.put(iFather, rank.get(jFather) + rank.get(iFather));
        } else {
            father.put(iFather, jFather);
            rank.put(jFather, rank.get(jFather) + rank.get(iFather));
        }
        return true;
    }

    public boolean isConnected(T i, T j) {
        if (!father.containsKey(i) || !father.containsKey(j)) return false;
        return find(i) == find(j);
    }

    public Map<T, Set<T>> getAllGroups() {
        Map<T, Set<T>> result = new HashMap<>();
        // 找出所有根
        for (T i : father.keySet()) {
            T f = find(i);
            result.putIfAbsent(f, new HashSet<>());
            result.get(f).add(i);
        }
        return result;
    }

    public int getNumOfGroups() {
        Set<T> s = new HashSet<T>();
        for (T i : father.keySet()) {
            s.add(find(i));
        }
        return s.size();
    }

    public boolean contains(T i) {
        return father.containsKey(i);
    }

}

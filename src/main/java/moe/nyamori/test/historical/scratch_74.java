package moe.nyamori.test.historical;

import java.util.*;

class scratch_74 {

    public static Integer[][][] memo;
    public static boolean[][] mtx;
    public static boolean[] visited;
    public static int result = 0;
    public static List<Integer> resultPath = new ArrayList<>();
    public static int[] furthestDistance;
    public static Map<String, Integer> nodeIdxMap = new HashMap<>();
    public static Map<Integer, String> idxNodeMap = new HashMap<>();
    public static Map<Integer, List<String>> idxLineMap = new HashMap<>();
    public static Map<Integer, List<Integer>> furthestPathMap = new HashMap<>();

    public static void main(String[] args) {
        var METRO_MAP = METRO_MAP_SZ_GAODE;
        int counter = 0;
        for (Map.Entry<String, String[]> e : METRO_MAP.entrySet()) {
            for (String n : e.getValue()) {
                if (nodeIdxMap.containsKey(n)) {
                    int id = nodeIdxMap.get(n);
                    idxLineMap.get(id).add(e.getKey());
                    continue;
                }
                idxNodeMap.put(counter, n);
                idxLineMap.putIfAbsent(counter, new ArrayList<>());
                idxLineMap.get(counter).add(e.getKey());
                nodeIdxMap.put(n, counter++);
            }
        }

        mtx = new boolean[counter][counter];
        visited = new boolean[counter];
        furthestDistance = new int[counter];

        for (Map.Entry<String, String[]> e : METRO_MAP.entrySet()) {
            String[] nodeArr = e.getValue();
            int len = nodeArr.length;
            for (int j = 1; j < len; j++) {
                int prevIdx = nodeIdxMap.get(nodeArr[j - 1]),
                        curIdx = nodeIdxMap.get(nodeArr[j]);
                mtx[prevIdx][curIdx] = true;
                mtx[curIdx][prevIdx] = true;
            }
        }

        long timing = System.currentTimeMillis();
        outer:
        for (int depth = 148; depth >= 3; depth--) {
            middle:
            for (int order = 0; order < counter; order++) {
                int startPoint = (order /*+ nodeIdxMap.get("广州南站")*/) % counter;
                int edgeCount = 0;
                boolean judge = false;
                inner:
                for (int sta = 0; sta < counter; sta++) {
                    if (mtx[startPoint][sta]) edgeCount++;
                    if (edgeCount > 2) { // 换乘站至少有三条边
                        judge = true;
                        break inner;
                    }
                }
                if (!judge) continue middle;

                System.err.println("+" + (System.currentTimeMillis() - timing) + "ms, " + depth + " depth, start from " + startPoint + "(" + idxNodeMap.get(startPoint) + ")");
                Arrays.fill(visited, false);
                List<Integer> path = new ArrayList<>();
                path.add(startPoint);
                visited[startPoint] = true;
                if (helper(startPoint, path, depth - 1)) break outer;
            }
        }
        System.err.println(resultPath.size());
        System.err.println(resultPath);
        for (int j : resultPath) {
            String output = idxNodeMap.get(j);
            if (idxLineMap.get(j).size() > 1) {
                output += "(" + idxLineMap.get(j) + ")";
            }
            System.err.println(output);
        }
        System.err.println("Timing: " + (System.currentTimeMillis() - timing) + "ms");

//
//        for (int i = 0; i < counter; i++) {
//            // 起点只能是换乘站, 简单判断一下
//            boolean judge = false;
//            int edgeCount = 0;
//            for (int j = 0; j < counter; j++) {
//                if (mtx[i][j]) edgeCount++;
//                if (edgeCount > 2) { // 换乘站至少有三条边
//                    judge = true;
//                    break;
//                }
//            }
//            if (!judge) {
//                continue;
//            }
//
//            if (!visited[i]) {
//                ArrayList<Integer> path = new ArrayList<>();
//                path.add(i);
//                helper(i, -1, 1, path);
//            }
//        }
//        System.err.println(result);
//        for (int j : resultPath) {
//            System.err.println(idxNodeMap.get(j));
//        }
    }


    private static boolean helper(int cur, List<Integer> path, int remain) {
        if (remain == 0) {
            if (mtx[cur][path.get(0)]) {
                resultPath = new ArrayList<>(path);
                return true;
            }
            return false;
        }

        for (int i = 0; i < mtx.length; i++) {
            if (!visited[i] && mtx[cur][i]) {
                visited[i] = true;
                path.add(i);
                boolean result = helper(i, path, remain - 1);
                path.remove(path.size() - 1);
                visited[i] = false;
                if (result) {
                    return true;
                }
            }
        }
        return false;
    }

//    private static void helper(int cur, int from, int len, List<Integer> path) {
//        if (visited[cur]) return;
//        visited[cur] = true;
//        furthestPathMap.put(cur, path);
//        furthestDistance[cur] = len;
//        for (int next = 0; next < mtx.length; next++) {
//            if (next != from && next != cur && mtx[cur][next]) {
//                List<Integer> nextPath = new ArrayList<>(path);
//                nextPath.add(next);
//                if (visited[next]) { // 成环
//                    int distance = furthestDistance[next] + furthestDistance[cur] - 1;
//                    if (distance > result) {
//                        result = distance;
//                        List<Integer> half = new ArrayList<>(path);
//                        List<Integer> nextHalf = new ArrayList<>(furthestPathMap.get(next));
//                        Collections.reverse(nextHalf);
//                        half.addAll(nextHalf);
//                        resultPath = half;
//                    }
//                } else { // 未成环 继续搜索
//                    helper(next, cur, len + 1, nextPath);
//                }
//            }
//        }
//    }

    public static final Map<String, String[]> METRO_MAP_GZ_GAODE = new HashMap<String, String[]>() {{
        put("1号线", new String[]{
                "西塱",
                "坑口",
                "花地湾",
                "芳村",
                "黄沙",
                "长寿路",
                "陈家祠",
                "西门口",
                "公园前",
                "农讲所",
                "烈士陵园",
                "东山口",
                "杨箕",
                "体育西路",
                "体育中心",
                "广州东站"
        });
        put("佛山2号线", new String[]{
                "广州南站",
                "林岳东(2号线)",
                "林岳西",
                "石洲",
                "仙涌",
                "花卉世界",
                "登洲",
                "湾华",
                "石梁",
                "魁奇路",
                "沙岗",
                "石湾",
                "张槎",
                "智慧新城",
                "绿岛湖",
                "湖涌",
                "南庄"
        });
        put("2号线", new String[]{
                "广州南站",
                "石壁",
                "会江",
                "南浦",
                "洛溪",
                "南洲",
                "东晓南",
                "江泰路",
                "昌岗",
                "江南西",
                "市二宫",
                "海珠广场",
                "公园前",
                "纪念堂",
                "越秀公园",
                "广州火车站",
                "三元里",
                "飞翔公园",
                "白云公园",
                "白云文化广场",
                "萧岗",
                "江夏",
                "黄边",
                "嘉禾望岗"
        });
        put("3号线", new String[]{
                "番禺广场",
                "市桥",
                "汉溪长隆",
                "大石",
                "厦滘",
                "沥滘",
                "大塘",
                "客村",
                "广州塔",
                "珠江新城",
                "体育西路",
                "石牌桥",
                "岗顶",
                "华师",
                "五山",
                "天河客运站"
        });
        put("3号线(北延段)", new String[]{
                "机场北(2号航站楼)",
                "机场南(1号航站楼)",
                "高增",
                "人和",
                "龙归",
                "嘉禾望岗",
                "白云大道北",
                "永泰",
                "同和",
                "京溪南方医院",
                "梅花园",
                "燕塘",
                "广州东站",
                "林和西",
                "体育西路"
        });
        put("4号线", new String[]{
                "黄村",
                "车陂",
                "车陂南",
                "万胜围",
                "官洲",
                "大学城北",
                "大学城南",
                "新造",
                "石碁",
                "海傍",
                "低涌",
                "东涌",
                "庆盛",
                "黄阁汽车城",
                "黄阁",
                "蕉门",
                "金洲",
                "飞沙角",
                "广隆",
                "大涌",
                "塘坑",
                "南横",
                "南沙客运港"
        });
        put("5号线", new String[]{
                "滘口",
                "坦尾",
                "中山八",
                "西场",
                "西村",
                "广州火车站",
                "小北",
                "淘金",
                "区庄",
                "动物园",
                "杨箕",
                "五羊邨",
                "珠江新城",
                "猎德",
                "潭村",
                "员村",
                "科韵路",
                "车陂南",
                "东圃",
                "三溪",
                "鱼珠",
                "大沙地",
                "大沙东",
                "文冲"
        });
        put("6号线", new String[]{
                "香雪",
                "萝岗",
                "苏元",
                "暹岗",
                "金峰",
                "黄陂",
                "高塘石",
                "柯木塱",
                "龙洞",
                "植物园",
                "长湴",
                "天河客运站",
                "燕塘",
                "天平架",
                "沙河顶",
                "黄花岗",
                "区庄",
                "东山口",
                "东湖",
                "团一大广场",
                "北京路",
                "海珠广场",
                "一德路",
                "文化公园",
                "黄沙",
                "如意坊",
                "坦尾",
                "河沙",
                "沙贝",
                "横沙",
                "浔峰岗"
        });
        put("7号线", new String[]{
                "大学城南",
                "板桥",
                "员岗",
                "南村万博",
                "汉溪长隆",
                "钟村",
                "谢村",
                "石壁",
                "广州南站",
                "大洲",
                "陈村北",
                "陈村",
                "锦龙",
                "南涌",
                "美的",
                "北滘公园",
                "美的大道"
        });
        put("8号线", new String[]{
                "滘心",
                "亭岗",
                "石井",
                "小坪",
                "石潭",
                "聚龙",
                "上步",
                "同德",
                "鹅掌坦",
                "彩虹桥",
                "陈家祠",
                "华林寺",
                "文化公园",
                "同福西",
                "凤凰新村",
                "沙园",
                "宝岗大道",
                "昌岗",
                "晓港",
                "中大",
                "鹭江",
                "客村",
                "赤岗",
                "磨碟沙",
                "新港东",
                "琶洲",
                "万胜围"
        });
        put("9号线", new String[]{
                "高增",
                "清塘",
                "清布",
                "莲塘",
                "马鞍山公园",
                "花都广场",
                "花果山公园",
                "花城路",
                "广州北站",
                "花都汽车城",
                "飞鹅岭"
        });
        put("13号线", new String[]{
                "鱼珠",
                "裕丰围",
                "双岗",
                "南海神庙",
                "夏园",
                "南岗",
                "沙村",
                "白江",
                "新塘",
                "官湖",
                "新沙"
        });
        put("14号线", new String[]{
                "嘉禾望岗",
                "白云东平",
                "夏良",
                "太和",
                "竹料",
                "钟落潭",
                "马沥",
                "新和",
                "太平",
                "神岗",
                "赤草",
                "从化客运站",
                "东风"
        });
        put("14号线支线(知识城线)", new String[]{
                "新和",
                "红卫",
                "新南",
                "枫下",
                "知识城",
                "何棠下",
                "旺村",
                "汤村",
                "镇龙北",
                "镇龙"
        });
        put("18号线", new String[]{
                "冼村",
                "磨碟沙",
                "龙潭",
                "沙溪",
                "南村万博",
                "番禺广场",
                "横沥",
                "万顷沙"
        });
        put("21号线", new String[]{
                "员村",
                "天河公园",
                "棠东",
                "黄村",
                "大观南路",
                "天河智慧城",
                "神舟路",
                "科学城",
                "苏元",
                "水西",
                "长平",
                "金坑",
                "镇龙西",
                "镇龙",
                "中新",
                "坑贝",
                "凤岗",
                "朱村",
                "山田",
                "钟岗",
                "增城广场"
        });
        put("22号线", new String[]{
                "陈头岗",
                "广州南站",
                "市广路",
                "番禺广场"
        });
        put("APM线", new String[]{
                "广州塔",
                "海心沙",
                "大剧院",
                "花城大道",
                "妇儿中心",
                "黄埔大道",
                "天河南",
                "体育中心南",
                "林和西"
        });
        put("广佛线", new String[]{
                "沥滘",
                "南洲",
                "石溪",
                "燕岗",
                "沙园",
                "沙涌",
                "鹤洞",
                "西塱",
                "菊树",
                "龙溪",
                "金融高新区",
                "千灯湖",
                "礌岗",
                "南桂路",
                "桂城",
                "朝安",
                "普君北路",
                "祖庙",
                "同济路",
                "季华园",
                "魁奇路",
                "澜石",
                "世纪莲",
                "东平",
                "新城东"
        });
    }};

    public static final Map<String, String[]> METRO_MAP_SZ_SOGOU = new HashMap<String, String[]>() {{
        put("1号线(罗宝线)", new String[]{
                "机场东",
                "后瑞",
                "固戍",
                "西乡",
                "坪洲",
                "宝体",
                "宝安中心",
                "新安",
                "前海湾",
                "鲤鱼门",
                "大新",
                "桃园",
                "深大",
                "高新园",
                "白石洲",
                "世界之窗",
                "华侨城",
                "侨城东",
                "竹子林",
                "车公庙",
                "香蜜湖",
                "购物公园",
                "会展中心",
                "岗厦",
                "华强路",
                "科学馆",
                "大剧院",
                "老街",
                "国贸",
                "罗湖"
        });
        put("2号线(蛇口线)", new String[]{
                "赤湾",
                "蛇口港",
                "海上世界",
                "水湾",
                "东角头",
                "湾厦",
                "海月",
                "登良",
                "后海",
                "科苑",
                "红树湾",
                "世界之窗",
                "侨城北",
                "深康",
                "安托山",
                "侨香",
                "香蜜",
                "香梅北",
                "景田",
                "莲花西",
                "福田",
                "市民中心",
                "岗厦北",
                "华强北",
                "燕南",
                "大剧院",
                "湖贝",
                "黄贝岭",
                "新秀",
                "莲塘口岸",
                "仙湖路",
                "莲塘"
        });
        put("8号线(盐田线)", new String[]{
                "莲塘",
                "梧桐山南",
                "沙头角",
                "海山",
                "盐田港西",
                "深外高中",
                "盐田路"
        });
        put("3号线(龙岗线)", new String[]{
                "双龙",
                "南联",
                "龙城广场",
                "吉祥",
                "爱联",
                "大运",
                "荷坳",
                "永湖",
                "横岗",
                "塘坑",
                "六约",
                "丹竹头",
                "大芬",
                "木棉湾",
                "布吉",
                "草埔",
                "水贝",
                "田贝",
                "翠竹",
                "晒布",
                "老街",
                "红岭",
                "通新岭",
                "华新",
                "莲花村",
                "少年宫",

                "福田",
                "购物公园",
                "石厦",
                "益田",
                "福保"
        });
        put("4号线(龙华线)", new String[]{
                "牛湖",
                "观澜湖",
                "松元厦",

                "观澜",
                "长湖",
                "茜坑",
                "竹村",
                "清湖北",
                "清湖",
                "龙华",
                "龙胜",
                "上塘",
                "红山",
                "深圳北站",
                "白石龙",
                "民乐",
                "上梅林",
                "莲花北",
                "少年宫",
                "市民中心",
                "会展中心",
                "福民",
                "福田口岸"
        });
        put("5号线(环中线)", new String[]{
                "黄贝岭",
                "怡景",
                "太安",
                "布心",
                "百鸽笼",
                "布吉",
                "长龙",
                "下水径",
                "上水径",
                "杨美",
                "坂田",
                "五和",
                "民治",
                "深圳北站",
                "长岭陂",
                "塘朗",
                "大学城",
                "西丽",
                "留仙洞",
                "兴东",
                "洪浪北",
                "灵芝",
                "翻身",
                "宝安中心",
                "宝华",
                "临海",
                "前海湾",
                "桂湾",
                "前湾",
                "前湾公园",
                "妈湾",
                "铁路公园",
                "荔湾",
                "赤湾"
        });
        put("7号线(西丽线)", new String[]{
                "西丽湖",
                "西丽",
                "茶光",


                "珠光",
                "龙井",
                "桃源村",
                "深云",
                "安托山",

                "农林",
                "车公庙",


                "上沙",
                "沙尾",
                "石厦",
                "皇岗村",
                "福民",

                "皇岗口岸",
                "福邻",

                "赤尾",
                "华强南",
                "华强北",
                "华新",

                "黄木岗",

                "八卦岭",
                "红岭北",
                "笋岗",
                "洪湖",
                "田贝",
                "太安"
        });
        put("9号线(梅林线)", new String[]{
                "前湾",
                "梦海",
                "怡海",
                "荔林",
                "南油西",
                "南油",
                "南山书城",
                "深大南",
                "粤海门",
                "高新南",

                "红树湾南",

                "深湾",
                "深圳湾公园",
                "下沙",

                "车公庙",
                "香梅",
                "景田",
                "梅景",

                "下梅林",
                "梅村",
                "上梅林",

                "孖岭",

                "银湖",
                "泥岗",

                "红岭北",
                "园岭",
                "红岭",
                "红岭南",

                "鹿丹村",
                "人民南",

                "向西村",
                "文锦"
        });
        put("11号线(机场线)", new String[]{
                "福田",


                "车公庙",


                "红树湾南",

                "后海",

                "南山",
                "前海湾",
                "宝安",
                "碧海湾",
                "机场",
                "机场北",
                "福永",
                "桥头",
                "塘尾",
                "马安山",
                "沙井",
                "后亭",
                "松岗",
                "碧头"
        });
        put("6号线(光明线)", new String[]{
                "松岗",
                "溪头",
                "松岗公园",
                "薯田埔",
                "合水口",
                "公明广场",
                "红花山",
                "楼村",
                "科学公园",
                "光明",
                "光明大街",
                "凤凰城",
                "长圳",
                "上屋",
                "官田",
                "阳台山东",
                "元芬",
                "上芬",
                "红山",
                "深圳北站",
                "梅林关",
                "翰岭",
                "银湖",
                "八卦岭",
                "体育中心",
                "通新岭",
                "科学馆"
        });
        put("10号线(坂田线)", new String[]{
                "双拥街",
                "平湖",
                "禾花",
                "华南城",
                "木古",
                "上李朗",
                "凉帽山",
                "甘坑",
                "雪象",
                "岗头",
                "华为",
                "贝尔路",
                "坂田北",
                "五和",
                "光雅园",
                "南坑",
                "雅宝",
                "孖岭",
                "冬瓜岭",
                "莲花村",
                "岗厦北",
                "岗厦",
                "福民",
                "福田口岸"});
    }};

    public final static Map<String, String[]> METRO_MAP_GZ_SOGOU = new HashMap<String, String[]>() {{
        put("1号线", new String[]{
                "西塱",
                "坑口",
                "花地湾",
                "芳村",
                "黄沙",
                "长寿路",
                "陈家祠",
                "西门口",
                "公园前",
                "农讲所",
                "烈士陵园",
                "东山口",
                "杨箕",
                "体育西路",
                "体育中心",
                "广州东站"});
        put("2号线", new String[]{
                "嘉禾望岗",
                "黄边",
                "江夏",
                "萧岗",
                "白云文化广场",
                "白云公园",
                "飞翔公园",
                "三元里",
                "广州火车站",
                "越秀公园",
                "纪念堂",
                "公园前",
                "海珠广场",
                "市二宫",
                "江南西",
                "昌岗",
                "江泰路",
                "东晓南",
                "南洲",
                "洛溪",
                "南浦",
                "会江",
                "石壁",
                "广州南站"});
        put("3号线", new String[]{
                "天河客运站",
                "五山",
                "华师",
                "岗顶",
                "石牌桥",
                "体育西路",
                "珠江新城",
                "广州塔",
                "客村",
                "大塘",
                "沥滘",
                "厦滘",
                "大石",
                "汉溪长隆",
                "市桥",
                "番禺广场"});
        put("3号线北延段", new String[]{
                "体育西路",
                "林和西",
                "广州东站",
                "燕塘",
                "梅花园",
                "京溪南方医院",
                "同和",
                "永泰",
                "白云大道北",
                "嘉禾望岗",
                "龙归",
                "人和",
                "高增",
                "机场南",
                "机场北"});
        put("4号线", new String[]{
                "黄村",
                "车陂",
                "车陂南",
                "万胜围",
                "官洲",
                "大学城北",
                "大学城南",
                "新造",
                "石碁",
                "海傍",
                "低涌",
                "东涌",
                "庆盛",
                "黄阁汽车城",
                "黄阁",
                "蕉门",
                "金洲",
                "飞沙角",
                "广隆",
                "大涌",
                "塘坑",
                "南横",
                "南沙客运港"});
        put("5号线", new String[]{
                "滘口",
                "坦尾",
                "中山八",
                "西场",
                "西村",
                "广州火车站",
                "小北",
                "淘金",
                "区庄",
                "动物园",
                "杨箕",
                "五羊邨",
                "珠江新城",
                "猎德",
                "潭村",
                "员村",
                "科韵路",
                "车陂南",
                "东圃",
                "三溪",
                "鱼珠",
                "大沙地",
                "大沙东",
                "文冲"});
        put("6号线", new String[]{
                "香雪",
                "萝岗",
                "苏元",
                "暹岗",
                "金峰",
                "黄陂",
                "高塘石",
                "柯木塱",
                "龙洞",
                "植物园",
                "长湴",
                "天河客运站",
                "燕塘",
                "天平架",
                "沙河顶",
                "黄花岗",
                "区庄",
                "东山口",
                "东湖",
                "团一大广场",
                "北京路",
                "海珠广场",
                "一德路",
                "文化公园",
                "黄沙",
                "如意坊",
                "坦尾",
                "河沙",
                "沙贝",
                "横沙",
                "浔峰岗"});
        put("7号线", new String[]{
                "广州南站",
                "石壁",
                "谢村",
                "钟村",
                "汉溪长隆",
                "南村万博",
                "员岗",
                "板桥",
                "大学城南"});
        put("8号线", new String[]{
                "滘心",
                "亭岗",
                "石井",
                "小坪",
                "石潭",
                "聚龙",
                "上步",
                "同德",
                "鹅掌坦",
                "陈家祠",
                "华林寺",
                "文化公园",
                "同福西",
                "凤凰新村",
                "沙园",
                "宝岗大道",
                "昌岗",
                "晓港",
                "中大",
                "鹭江",
                "客村",
                "赤岗",
                "磨碟沙",
                "新港东",
                "琶洲",
                "万胜围"});
        put("9号线,", new String[]{
                "高增",
                "清塘",
                "清布",
                "莲塘",
                "马鞍山公园",
                "花都广场",
                "花果山公园",
                "花城路",
                "广州北站",
                "花都汽车城",
                "飞鹅岭"});
        put("13号线", new String[]{
                "鱼珠",
                "裕丰围",
                "双岗",
                "南海神庙",
                "夏园",
                "南岗",
                "沙村",
                "白江",
                "新塘",
                "官湖",
                "新沙"});
        put("14号线支线", new String[]{
                "新和",
                "红卫",
                "新南",
                "枫下",
                "知识城",
                "何棠下",
                "旺村",
                "汤村",
                "镇龙北",
                "镇龙"});
        put("广佛线", new String[]{
                "沥滘",
                "南洲",
                "石溪",
                "燕岗",
                "沙园",
                "沙涌",
                "鹤洞",
                "西塱",
                "菊树",
                "龙溪",
                "金融高新区",
                "千灯湖",
                "礌岗",
                "南桂路",
                "桂城",
                "朝安",
                "普君北路",
                "祖庙",
                "同济路",
                "季华园",
                "魁奇路",
                "澜石",
                "世纪莲",
                "东平",
                "新城东"});
        put("APM线", new String[]{
                "林和西",
                "体育中心南",
                "天河南",
                "黄埔大道",
                "妇儿中心",
                "花城大道",
                "大剧院",
                "海心沙",
                "广州塔"});
        put("14号线", new String[]{
                "嘉禾望岗",
                "白云东平",
                "夏良",
                "太和",
                "竹料",
                "钟落潭",
                "马沥",
                "新和",
                "太平",
                "神岗",
                "赤草",
                "从化客运站",
                "东风"});
        put("21号线", new String[]{
                "员村",
                "天河公园",
                "棠东",
                "黄村",
                "大观南路",
                "天河智慧城",
                "神舟路",
                "科学城",
                "苏元",
                "水西",
                "长平",
                "金坑",
                "镇龙西",
                "镇龙",
                "中新",
                "坑贝",
                "凤岗",
                "朱村",
                "山田",
                "钟岗",
                "增城广场"});
    }};

    public static final Map<String, String[]> METRO_MAP_SZ_GAODE = new HashMap<String, String[]>() {{
        put("1号线/罗宝线", new String[]{
                "罗湖",
                "国贸",
                "老街",
                "大剧院",
                "大剧院",
                "科学馆",
                "华强路",
                "岗厦",
                "会展中心",
                "购物公园",
                "香蜜湖",
                "车公庙",
                "竹子林",
                "侨城东",
                "华侨城",
                "世界之窗",
                "白石洲",
                "高新园",
                "深大",
                "桃园",
                "大新",
                "鲤鱼门",
                "前海湾",
                "新安",
                "宝安中心",
                "宝体",
                "坪洲",
                "西乡",
                "固戍",
                "后瑞",
                "机场东"
        });

        put("2号线/8号线", new String[]{
                "赤湾",
                "蛇口港",
                "海上世界",
                "水湾",
                "东角头",
                "湾厦",
                "海月",
                "登良",
                "后海",
                "科苑",
                "红树湾",
                "世界之窗",
                "侨城北",
                "深康",
                "安托山",
                "侨香",
                "香蜜",
                "香梅北",
                "景田",
                "莲花西",
                "福田",
                "市民中心",
                "岗厦北",
                "华强北",
                "燕南",
                "大剧院",
                "湖贝",
                "黄贝岭",
                "新秀",
                "莲塘口岸",
                "仙湖路",
                "莲塘",
                "梧桐山南",
                "沙头角",
                "海山",
                "盐田港西",
                "深外高中",
                "盐田路"
        });

        put("3号线/龙岗线", new String[]{
                "福保",
                "益田",
                "石厦",
                "购物公园",
                "福田",
                "少年宫",
                "莲花村",
                "华新",
                "通新岭",
                "红岭",
                "老街",
                "晒布",
                "翠竹",
                "田贝",
                "水贝",
                "草埔",
                "布吉",
                "木棉湾",
                "大芬",
                "丹竹头",
                "六约",
                "塘坑",
                "横岗",
                "永湖",
                "荷坳",
                "大运",
                "爱联",
                "吉祥",
                "龙城广场",
                "南联",
                "双龙"
        });

        put("4号线/龙华线", new String[]{
                "福田口岸",
                "福民",
                "会展中心",
                "市民中心",
                "少年宫",
                "莲花北",
                "上梅林",
                "民乐",
                "白石龙",
                "深圳北站",
                "红山",
                "上塘",
                "龙胜",
                "龙华",
                "清湖",
                "清湖北",
                "竹村",
                "茜坑",
                "长湖",
                "观澜",
                "松元厦",
                "观澜湖",
                "牛湖"
        });

        put("5号线/环中线", new String[]{
                "赤湾",
                "荔湾",
                "铁路公园",
                "妈湾",
                "前湾公园",
                "前湾",
                "桂湾",
                "前海湾",
                "临海",
                "宝华",
                "宝安中心",
                "翻身",
                "灵芝",
                "洪浪北",
                "兴东",
                "留仙洞",
                "西丽",
                "大学城",
                "塘朗",
                "长岭陂",
                "深圳北站",
                "民治",
                "五和",
                "坂田",
                "杨美",
                "上水径",
                "下水径",
                "长龙",
                "布吉",
                "百鸽笼",
                "布心",
                "太安",
                "怡景",
                "黄贝岭"
        });

        put("6号线/光明线", new String[]{
                "松岗",
                "溪头",
                "松岗公园",
                "薯田埔",
                "合水口",
                "公明广场",
                "红花山",
                "楼村",
                "科学公园",
                "光明",
                "光明大街",
                "凤凰城",
                "长圳",
                "上屋",
                "官田",
                "阳台山东",
                "元芬",
                "上芬",
                "红山",
                "深圳北站",
                "梅林关",
                "翰岭",
                "银湖",
                "八卦岭",
                "体育中心",
                "通新岭",
                "科学馆"
        });

        put("6号线支线", new String[]{
                "光明",
                "圳美",
                "中大",
                "深理工"
        });

        put("7号线/西丽线", new String[]{
                "西丽湖",
                "西丽",
                "茶光",
                "珠光",
                "龙井",
                "桃源村",
                "深云",
                "安托山",
                "农林",
                "车公庙",
                "上沙",
                "沙尾",
                "石厦",
                "皇岗村",
                "福民",
                "皇岗口岸",
                "赤尾",
                "华强南",
                "华强北",
                "华新",
                "黄木岗",
                "八卦岭",
                "红岭北",
                "笋岗",
                "洪湖",
                "田贝",
                "太安"
        });

        put("9号线/梅林线", new String[]{
                "前湾",
                "梦海",
                "怡海",
                "荔林",
                "南油西",
                "南油",
                "南山书城",
                "深大南",
                "粤海门",
                "高新南",
                "红树湾南",
                "深湾",
                "深圳湾公园",
                "下沙",
                "车公庙",
                "香梅",
                "景田",
                "梅景",
                "下梅林",
                "梅村",
                "上梅林",
                "孖岭",
                "银湖",
                "泥岗",
                "红岭北",
                "园岭",
                "红岭",
                "红岭南",
                "鹿丹村",
                "人民南",
                "向西村",
                "文锦"
        });

        put("10号线/坂田线", new String[]{
                "双拥街",
                "平湖",
                "禾花",
                "华南城",
                "木古",
                "上李朗",
                "凉帽山",
                "甘坑",
                "雪象",
                "岗头",
                "华为",
                "贝尔路",
                "坂田北",
                "五和",
                "光雅园",
                "南坑",
                "雅宝",
                "孖岭",
                "冬瓜岭",
                "莲花村",
                "岗厦北",
                "岗厦",
                "福民",
                "福田口岸"
        });

        put("11号线/机场线", new String[]{
                "碧头",
                "松岗",
                "后亭",
                "沙井",
                "马安山",
                "塘尾",
                "桥头",
                "福永",
                "机场北",
                "机场",
                "碧海湾",
                "宝安",
                "前海湾",
                "南山",
                "后海",
                "红树湾南",
                "车公庙",
                "福田",
                "岗厦北"
        });

        put("12号线/南宝线", new String[]{
                "海上田园东",
                "海上田园南",
                "国展北",
                "国展",
                "福海西",
                "桥头西",
                "福永",
                "怀德",
                "福围",
                "机场东",
                "兴围",
                "黄田",
                "钟屋南",
                "西乡桃源",
                "平峦山",
                "宝田一路",
                "宝安客运站",
                "流塘",
                "上川",
                "灵芝",
                "新安公园",
                "同乐南",
                "中山公园",
                "南头古城",
                "桃园",
                "南山",
                "南光",
                "南油",
                "四海",
                "花果山",
                "海上世界",
                "太子湾",
                "左炮台东"
        });

        put("14号线/东部快线", new String[]{
                "岗厦北",
                "黄木岗",
                "罗湖北",
                "布吉",
                "石芽岭",
                "六约北",
                "四联",
                "坳背",
                "大运",
                "嶂背",
                "南约",
                "宝龙",
                "锦龙",
                "坪山围",
                "坪山广场",
                "坪山中心",
                "坑梓",
                "沙田"
        });

        put("20号线", new String[]{
                "机场北",
                "国展南",
                "国展",
                "国展北",
                "会展城"
        });
    }};
}
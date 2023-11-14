package moe.nyamori.test.ordered._1300

import kotlin.math.min

object LC1334 {
    // 1334. 阈值距离内邻居最少的城市
    // 中等
    // 有 n 个城市，按从 0 到 n-1 编号。给你一个边数组 edges，其中 edges[i] = [fromi, toi, weighti] 代表 fromi 和 toi 两个城市之间的双向加权边，距离阈值是一个整数 distanceThreshold。
    //
    // 返回能通过某些路径到达其他城市数目最少、且路径距离 最大 为 distanceThreshold 的城市。如果有多个这样的城市，则返回编号最大的城市。
    //
    // 注意，连接城市 i 和 j 的路径的距离等于沿该路径的所有边的权重之和。
    // 2 <= n <= 100
    // 1 <= edges.length <= n * (n - 1) / 2
    // edges[i].length == 3
    // 0 <= fromi < toi < n
    // 1 <= weighti, distanceThreshold <= 10^4
    // 所有 (fromi, toi) 都是不同的。
    fun findTheCity(n: Int, edges: Array<IntArray>, distanceThreshold: Int) = run {
        val inf = Integer.MAX_VALUE shr 1
        val distances = Array(n) { IntArray(n) { inf } }
        edges.forEach { (u, v, w) ->
            distances[u][v] = w
            distances[v][u] = w
        }
        (0..<n).forEach { i ->
            (0..<n).forEach { j ->
                (0..<n).forEach { k ->
                    min(distances[j][k], distances[j][i] + distances[i][k])
                        .also { distances[k][j] = it }
                        .also { distances[j][k] = it }
                }
            }
        }
        distances.indices.reversed().minBy { outerIdx ->
            distances[outerIdx].filterIndexed { innerIdx, v ->
                outerIdx != innerIdx && v <= distanceThreshold
            }.count()
        }
    }
}
"""
快速种子数据导入：从 Netease API 拉取热门歌曲写入 Neo4j
不需要本地音频文件，纯 API → Neo4j
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from retrieval.neo4j_client import get_neo4j_client

NETEASE_URL = "http://netease:3000"

# 要搜索的关键词（覆盖多种风格）
SEARCH_TERMS = [
    ("周杰伦", 10),
    ("Taylor Swift", 10),
    ("林俊杰", 10),
    ("邓紫棋", 10),
    ("陈奕迅", 10),
    ("久石让", 8),
    ("BTS", 8),
    ("Ed Sheeran", 8),
    ("Coldplay", 8),
    ("The Beatles", 8),
]

def search_netease(keyword: str, limit: int = 10) -> list:
    """搜索网易云歌曲"""
    try:
        resp = requests.get(f"{NETEASE_URL}/search", params={"keywords": keyword, "limit": limit}, timeout=10)
        data = resp.json()
        if data.get("code") == 200:
            return data["result"].get("songs", [])
    except Exception as e:
        print(f"  搜索 '{keyword}' 失败: {e}")
    return []


def seed_neo4j():
    client = get_neo4j_client()
    total = 0
    seen = set()

    for keyword, limit in SEARCH_TERMS:
        print(f"\n🔍 搜索: {keyword}")
        songs = search_netease(keyword, limit)

        for song in songs:
            song_id = str(song["id"])
            if song_id in seen:
                continue
            seen.add(song_id)

            title = song["name"]
            artists = [a["name"] for a in song.get("artists", []) if a.get("name")]
            album_name = song.get("album", {}).get("name", "")
            duration = song.get("duration", 0)
            artist_names = " / ".join(artists) if artists else "未知"

            try:
                client.execute_query("""
                    MERGE (s:Song {music_id: $music_id})
                    SET s.title = $title,
                        s.album = $album,
                        s.duration = $duration,
                        s.language = 'Unknown',
                        s.unplayable_stub = false

                    FOREACH (artist_name IN $artists |
                        MERGE (a:Artist {name: artist_name})
                        MERGE (s)-[:PERFORMED_BY]->(a)
                    )

                    FOREACH (_ IN CASE WHEN $album <> '' THEN [1] ELSE [] END |
                        MERGE (al:Album {name: $album})
                        MERGE (s)-[:BELONGS_TO]->(al)
                    )
                """, {
                    "music_id": song_id,
                    "title": title,
                    "album": album_name,
                    "duration": duration,
                    "artists": artists if artists else ["未知"],
                })
                total += 1
                print(f"  ✅ {title} — {artist_names}")
            except Exception as e:
                print(f"  ❌ {title} 写入失败: {e}")

    # 统计
    count = client.execute_query("MATCH (s:Song) RETURN count(s) AS c", {})
    artist_count = client.execute_query("MATCH (a:Artist) RETURN count(a) AS c", {})
    print(f"\n📊 导入完成：{count[0]['c']} 首歌, {artist_count[0]['c']} 位歌手")


if __name__ == "__main__":
    seed_neo4j()

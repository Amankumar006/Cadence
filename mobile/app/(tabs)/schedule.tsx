import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { FlashList } from "@shopify/flash-list";
import { useSchedule } from "../../hooks/useSchedule";
import { COLORS, TYPE_COLORS } from "../../constants/colors";
import { authService } from "../../services/firebase";

function getDayLabel(offset: number): string {
  const d = new Date();
  d.setDate(d.getDate() + offset);
  return d.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });
}

function getDateKey(offset: number): string {
  const d = new Date();
  d.setDate(d.getDate() + offset);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

export default function ScheduleScreen() {
  const [selectedOffset, setSelectedOffset] = useState(0);
  const dateKey = getDateKey(selectedOffset);
  const { schedule, loading, currentMode, toggleMode, updateBlockStatus } = useSchedule(dateKey);
  const [uid, setUid] = useState(authService.currentUser?.uid || "mock_user_123");

  useEffect(() => {
    const unsubAuth = authService.onAuthStateChanged((user: any) => {
      if (user) {
        setUid(user.uid);
      } else {
        setUid("mock_user_123");
      }
    });
    return unsubAuth;
  }, []);

  const daysOffset = [-3, -2, -1, 0, 1, 2, 3];

  const parsedDate = new Date(dateKey);
  const dayName = parsedDate.toLocaleDateString("en-US", { weekday: "long" });
  const dayDateStr = parsedDate.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  const completedCount = schedule.filter(b => b.completed).length;
  const blocksCount = schedule.length;

  const renderScheduleBlock = (item: any, index: number) => {
    const color = TYPE_COLORS[item.type as keyof typeof TYPE_COLORS] ?? COLORS.muted;
    const isCritical = item.type === "study" && item.title.toLowerCase().includes("critical");
    return (
      <View style={styles.tlBlock}>
        <Text style={styles.tlTime}>{item.time}</Text>
        <View style={[styles.tlBar, { backgroundColor: color }]} />
        <View style={[styles.tlCard, isCritical && { borderColor: "rgba(255,138,155,0.25)", backgroundColor: "rgba(255,138,155,0.04)" }]}>
          <View style={styles.tlCardContent}>
            <Text style={[styles.tlTitle, isCritical && { color: COLORS.red }]}>{item.title}</Text>
            {item.notes ? <Text style={styles.tlNote}>{item.notes}</Text> : null}
            
            <View style={styles.statusRow}>
              {item.completed ? (
                <TouchableOpacity style={styles.completedBadge} onPress={() => updateBlockStatus(index, "none")}>
                  <Text style={styles.completedBadgeText}>✓ Completed</Text>
                </TouchableOpacity>
              ) : item.skipped ? (
                <TouchableOpacity style={styles.skippedBadge} onPress={() => updateBlockStatus(index, "none")}>
                  <Text style={styles.skippedBadgeText}>✗ Skipped</Text>
                </TouchableOpacity>
              ) : (
                <View style={styles.actionButtons}>
                  <TouchableOpacity style={styles.doneBtn} onPress={() => updateBlockStatus(index, "completed")}>
                    <Text style={styles.doneBtnText}>✓ Done</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.skipBtn} onPress={() => updateBlockStatus(index, "skipped")}>
                    <Text style={styles.skipBtnText}>✗ Skip</Text>
                  </TouchableOpacity>
                </View>
              )}
            </View>
          </View>
          <Text style={styles.tlTag}>{item.duration_min}m</Text>
        </View>
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.container} edges={["left", "right"]}>
      {/* DAY STATS CARD */}
      <View style={styles.dayCard}>
        <View style={styles.dayCardTop}>
          <View>
            <Text style={styles.dayLabelName}>{dayName}</Text>
            <Text style={styles.dayDate}>{dayDateStr}</Text>
          </View>
          <View style={styles.modeBadge}>
            <Text style={styles.modeBadgeText}>
              {currentMode === "weekday" ? "💼 Weekday" : "🌴 Weekend"}
            </Text>
          </View>
        </View>
        
        <View style={styles.dayStats}>
          <View style={styles.dStat}>
            <Text style={styles.dStatVal}>{blocksCount}</Text>
            <Text style={styles.dStatLabel}>Blocks</Text>
          </View>
          <View style={styles.dStat}>
            <Text style={[styles.dStatVal, { color: COLORS.green }]}>{completedCount}</Text>
            <Text style={styles.dStatLabel}>Done</Text>
          </View>
          <View style={styles.dStat}>
            <TouchableOpacity onPress={toggleMode} style={styles.toggleBtn}>
              <Text style={styles.toggleBtnText}>Switch</Text>
            </TouchableOpacity>
            <Text style={styles.dStatLabel}>Routine</Text>
          </View>
        </View>
      </View>

      {/* WEEK STRIP */}
      <View style={styles.weekStrip}>
        {daysOffset.map((offset) => {
          const d = new Date();
          d.setDate(d.getDate() + offset);
          const dayLetter = d.toLocaleDateString("en-US", { weekday: "short" })[0];
          const dayNum = d.getDate();
          const isActive = selectedOffset === offset;
          return (
            <TouchableOpacity
              key={offset}
              style={[styles.dayPill, isActive && styles.dayPillActive]}
              onPress={() => setSelectedOffset(offset)}
            >
              <Text style={[styles.dpLetter, isActive && styles.dpLetterActive]}>
                {dayLetter}
              </Text>
              <Text style={[styles.dpNum, isActive && styles.dpNumActive]}>
                {dayNum}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>

      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={COLORS.accent} />
          <Text style={styles.loadingText}>Building your schedule...</Text>
        </View>
      ) : (
        <FlashList
          data={schedule}
          renderItem={({ item, index }) => renderScheduleBlock(item, index)}
          keyExtractor={(item, index) => item.id || `${item.time}-${index}`}
          contentContainerStyle={styles.listContainer}
          ListEmptyComponent={
            <View style={styles.emptyContainer}>
              <Text style={styles.emptyText}>No schedule yet for {dayDateStr}.</Text>
              <Text style={styles.emptySubtext}>Ask your AI Co-pilot to generate one!</Text>
            </View>
          }
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.bg,
  },
  dayCard: {
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 22,
    padding: 16,
    margin: 16,
    marginBottom: 8,
  },
  dayCardTop: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: 16,
  },
  dayLabelName: {
    fontSize: 11,
    fontWeight: "700",
    color: COLORS.accent,
    textTransform: "uppercase",
    letterSpacing: 1,
  },
  dayDate: {
    fontSize: 18,
    fontWeight: "800",
    color: COLORS.text,
    marginTop: 2,
  },
  modeBadge: {
    backgroundColor: COLORS.panel,
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  modeBadgeText: {
    color: COLORS.textMuted,
    fontSize: 10,
    fontWeight: "600",
  },
  dayStats: {
    flexDirection: "row",
    gap: 8,
  },
  dStat: {
    flex: 1,
    backgroundColor: COLORS.panel,
    borderRadius: 12,
    paddingVertical: 10,
    alignItems: "center",
    justifyContent: "center",
  },
  dStatVal: {
    fontSize: 16,
    fontWeight: "800",
    color: COLORS.text,
  },
  dStatLabel: {
    fontSize: 8,
    color: COLORS.textDim,
    textTransform: "uppercase",
    fontWeight: "600",
    marginTop: 2,
    letterSpacing: 0.5,
  },
  toggleBtn: {
    paddingVertical: 1,
  },
  toggleBtnText: {
    fontSize: 11,
    fontWeight: "700",
    color: COLORS.accent,
  },
  weekStrip: {
    flexDirection: "row",
    paddingHorizontal: 16,
    gap: 6,
    marginBottom: 16,
  },
  dayPill: {
    flex: 1,
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 12,
    paddingVertical: 8,
    alignItems: "center",
    justifyContent: "center",
  },
  dayPillActive: {
    backgroundColor: COLORS.accentBg,
    borderColor: COLORS.accentLine,
  },
  dpLetter: {
    fontSize: 8,
    color: COLORS.textDim,
    textTransform: "uppercase",
    fontWeight: "600",
  },
  dpLetterActive: {
    color: COLORS.accent,
  },
  dpNum: {
    fontSize: 12,
    color: COLORS.textMuted,
    fontWeight: "700",
    marginTop: 2,
  },
  dpNumActive: {
    color: COLORS.accent,
  },
  listContainer: {
    paddingHorizontal: 16,
    paddingBottom: 40,
  },
  tlBlock: {
    flexDirection: "row",
    alignItems: "stretch",
    marginBottom: 10,
  },
  tlTime: {
    width: 44,
    fontSize: 11,
    color: COLORS.textDim,
    fontFamily: "monospace",
    paddingTop: 8,
    textAlign: "right",
    paddingRight: 8,
  },
  tlBar: {
    width: 3,
    borderRadius: 1.5,
    marginRight: 10,
  },
  tlCard: {
    flex: 1,
    flexDirection: "row",
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 12,
    padding: 12,
    alignItems: "center",
    justifyContent: "space-between",
  },
  tlCardContent: {
    flex: 1,
    marginRight: 8,
  },
  tlTitle: {
    fontSize: 13,
    fontWeight: "700",
    color: COLORS.text,
  },
  tlNote: {
    fontSize: 10,
    color: COLORS.textDim,
    marginTop: 2,
  },
  tlTag: {
    fontSize: 9,
    fontWeight: "700",
    backgroundColor: COLORS.panel,
    color: COLORS.textMuted,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 10,
    overflow: "hidden",
  },
  statusRow: {
    flexDirection: "row",
    marginTop: 8,
  },
  completedBadge: {
    backgroundColor: COLORS.accentBg,
    borderWidth: 1,
    borderColor: COLORS.accentLine,
    borderRadius: 12,
    paddingHorizontal: 8,
    paddingVertical: 2,
  },
  completedBadgeText: {
    color: COLORS.accent,
    fontSize: 9,
    fontWeight: "700",
  },
  skippedBadge: {
    backgroundColor: "rgba(255, 79, 109, 0.08)",
    borderWidth: 1,
    borderColor: "rgba(255, 79, 109, 0.2)",
    borderRadius: 12,
    paddingHorizontal: 8,
    paddingVertical: 2,
  },
  skippedBadgeText: {
    color: COLORS.red,
    fontSize: 9,
    fontWeight: "700",
  },
  actionButtons: {
    flexDirection: "row",
    gap: 6,
  },
  doneBtn: {
    borderWidth: 1,
    borderColor: COLORS.accent,
    backgroundColor: COLORS.accentBg,
    borderRadius: 12,
    paddingHorizontal: 10,
    paddingVertical: 2,
  },
  doneBtnText: {
    color: COLORS.accent,
    fontSize: 9,
    fontWeight: "700",
  },
  skipBtn: {
    borderWidth: 1,
    borderColor: COLORS.border,
    backgroundColor: COLORS.panel,
    borderRadius: 12,
    paddingHorizontal: 10,
    paddingVertical: 2,
  },
  skipBtnText: {
    color: COLORS.textMuted,
    fontSize: 9,
    fontWeight: "600",
  },
  loadingContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingVertical: 64,
  },
  loadingText: {
    color: COLORS.textDim,
    marginTop: 12,
    fontFamily: "monospace",
  },
  emptyContainer: {
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 64,
  },
  emptyText: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: "600",
  },
  emptySubtext: {
    color: COLORS.textDim,
    fontSize: 12,
    marginTop: 4,
  },
});

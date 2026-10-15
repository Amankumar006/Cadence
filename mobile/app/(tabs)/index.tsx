import React, { useState, useRef } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  Modal
} from "react-native";
import { SafeAreaView, useSafeAreaInsets } from "react-native-safe-area-context";
import { FlashList, type FlashListRef } from "@shopify/flash-list";
import { useAgentChat } from "../../hooks/useAgentChat";
import { COLORS, TYPE_COLORS } from "../../constants/colors";

export default function ChatScreen() {
  const insets = useSafeAreaInsets();
  const threadId = "default-chat-thread";
  const {
    messages,
    input,
    setInput,
    handleSubmit,
    approveSchedule,
    isApproving,
    isLoading,
    isOffline
  } = useAgentChat(threadId);

  const [editModalVisible, setEditModalVisible] = useState(false);
  const [selectedSchedule, setSelectedSchedule] = useState<any[]>([]);
  const [originalSchedule, setOriginalSchedule] = useState<any[]>([]);
  const [expandedBlockIndex, setExpandedBlockIndex] = useState<number | null>(null);
  const flatListRef = useRef<FlashListRef<any>>(null);

  const openEditModal = (schedule: any[]) => {
    setOriginalSchedule(schedule);
    setSelectedSchedule(JSON.parse(JSON.stringify(schedule))); // Deep copy
    setExpandedBlockIndex(null);
    setEditModalVisible(true);
  };

  const handleSaveEdits = () => {
    const edits: any[] = [];
    
    // Find removed blocks: original blocks that are not in selectedSchedule
    originalSchedule.forEach(orig => {
      const stillExists = selectedSchedule.some(sel => sel.title === orig.title);
      if (!stillExists) {
        edits.push({ action: "remove", block: orig.title });
      }
    });

    // Find moved blocks: selected blocks that exist in original blocks but have a different time
    selectedSchedule.forEach(sel => {
      const orig = originalSchedule.find(o => o.title === sel.title);
      if (orig && orig.time !== sel.time) {
        edits.push({ action: "move", block: sel.title, new_time: sel.time });
      }
    });

    approveSchedule({ approved: false, edits });
    setEditModalVisible(false);
  };

  const renderProposedSchedule = (schedule: any[]) => {
    return (
      <View style={styles.scheduleCard}>
        <View style={styles.scheduleHeaderRow}>
          <View style={styles.headerDot} />
          <Text style={styles.scheduleHeader}>Proposed Schedule · {schedule.length} blocks</Text>
        </View>
        {schedule.map((block: any, idx: number) => {
          const color = TYPE_COLORS[block.type as keyof typeof TYPE_COLORS] ?? COLORS.muted;
          const isCritical = block.type === "study" && block.title.toLowerCase().includes("critical");
          return (
            <View key={idx} style={styles.blockRow}>
              <View style={[styles.blockBar, { backgroundColor: color }]} />
              <Text style={styles.blockTime}>{block.time}</Text>
              <View style={styles.blockContent}>
                <Text style={[styles.blockTitle, isCritical && { color: COLORS.red }]}>{block.title}</Text>
                {block.notes ? <Text style={styles.blockNotes}>{block.notes}</Text> : null}
              </View>
              <Text style={styles.blockDuration}>{block.duration_min}m</Text>
            </View>
          );
        })}

        <View style={styles.actionRow}>
          <TouchableOpacity
            style={styles.approveButton}
            onPress={() => approveSchedule({ approved: true, edits: [] })}
            disabled={isApproving}
          >
            {isApproving ? (
              <ActivityIndicator size="small" color="#1a1611" />
            ) : (
              <Text style={styles.actionButtonText}>Looks good ✓</Text>
            )}
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.rejectButton}
            onPress={() => openEditModal(schedule)}
            disabled={isApproving}
          >
            <Text style={styles.rejectButtonText}>Edit</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  };

  const renderMessageItem = ({ item }: { item: any }) => {
    const isUser = item.role === "user";
    
    // Schedules arrive as a typed data-schedule part, never as JSON in the
    // text, so there is nothing to sniff out of `content`.
    const proposedSchedule: any[] | null =
      item.data && typeof item.data === "object" && item.data.schedule
        ? item.data.schedule
        : null;

    return (
      <View style={[styles.messageRow, isUser ? styles.userRow : styles.assistantRow]}>
        <View style={[styles.bubble, isUser ? styles.userBubble : styles.assistantBubble]}>
          {/* A message can carry both text and a schedule data part — show the
              text above the schedule rather than replacing it. */}
          {item.content ? (
            <Text style={[styles.messageText, isUser ? styles.userText : styles.assistantText]}>
              {item.content}
            </Text>
          ) : null}
          {proposedSchedule ? renderProposedSchedule(proposedSchedule) : null}
        </View>
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.container} edges={["top", "left", "right"]}>
      <KeyboardAvoidingView
        style={styles.container}
        behavior={Platform.OS === "ios" ? "padding" : undefined}
        keyboardVerticalOffset={Platform.OS === "ios" ? 88 : 0}
      >
        {isOffline && (
          <View style={styles.offlineBanner}>
            <Text style={styles.offlineBannerText}>Agent offline — check your connection</Text>
          </View>
        )}
        <FlashList
          ref={flatListRef}
          data={messages}
          renderItem={renderMessageItem}
          keyExtractor={(item, index) => item.id || index.toString()}
          contentContainerStyle={styles.messageList}
          onContentSizeChange={() => flatListRef.current?.scrollToEnd({ animated: true })}
        />

        {isLoading || isApproving ? (
          <View style={styles.loadingIndicator}>
            <ActivityIndicator size="small" color={COLORS.accent} />
            <Text style={styles.loadingText}>
              {isApproving ? "Updating schedule..." : "Agent is thinking..."}
            </Text>
          </View>
        ) : null}

        <View style={[
          styles.inputContainer,
          { paddingBottom: insets.bottom > 0 ? insets.bottom : 12 }
        ]}>
          <View style={styles.inputBar}>
            <TextInput
              style={styles.textInput}
              placeholder="Ask anything..."
              placeholderTextColor={COLORS.textDim}
              value={input}
              onChangeText={setInput}
              multiline={true}
              blurOnSubmit={false}
            />
            <TouchableOpacity style={styles.sendButton} onPress={handleSubmit}>
              <Text style={styles.sendButtonArrow}>↑</Text>
            </TouchableOpacity>
          </View>
        </View>
      </KeyboardAvoidingView>

      <Modal
        visible={editModalVisible}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setEditModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Edit Proposed Schedule</Text>
            <Text style={styles.modalSubtitle}>Tap a block to change its start time or remove it.</Text>
            
            <FlashList
              data={selectedSchedule}
              keyExtractor={(item, index) => `${item.title}-${index}`}
              contentContainerStyle={styles.modalList}
              renderItem={({ item, index }) => {
                const color = TYPE_COLORS[item.type as keyof typeof TYPE_COLORS] ?? COLORS.muted;
                const isExpanded = expandedBlockIndex === index;
                
                return (
                  <View style={styles.modalBlockContainer}>
                    <TouchableOpacity
                      style={[styles.modalBlockRow, { borderLeftColor: color }]}
                      onPress={() => setExpandedBlockIndex(isExpanded ? null : index)}
                    >
                      <Text style={styles.modalBlockTime}>{item.time}</Text>
                      <View style={styles.blockContent}>
                        <Text style={styles.blockTitle}>{item.title}</Text>
                        <Text style={styles.modalBlockType}>{item.type.toUpperCase()}</Text>
                      </View>
                      <Text style={styles.modalExpandIcon}>{isExpanded ? "▲" : "▼"}</Text>
                    </TouchableOpacity>
                    
                    {isExpanded && (
                      <View style={styles.expandedControls}>
                        <View style={styles.timeInputRow}>
                          <Text style={styles.inputLabel}>Start Time:</Text>
                          <TextInput
                            style={styles.timeInput}
                            value={item.time}
                            placeholder="HH:MM"
                            placeholderTextColor={COLORS.muted}
                            onChangeText={(text) => {
                              setSelectedSchedule(prev => prev.map((b, idx) => idx === index ? { ...b, time: text } : b));
                            }}
                          />
                        </View>
                        <TouchableOpacity
                          style={styles.removeBlockButton}
                          onPress={() => {
                            setSelectedSchedule(prev => prev.filter((_, idx) => idx !== index));
                            setExpandedBlockIndex(null);
                          }}
                        >
                          <Text style={styles.removeBlockText}>Remove Block</Text>
                        </TouchableOpacity>
                      </View>
                    )}
                  </View>
                );
              }}
            />
            
            <View style={styles.modalActionRow}>
              <TouchableOpacity style={styles.modalSaveButton} onPress={handleSaveEdits}>
                <Text style={styles.modalSaveText}>Save Changes</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.modalCancelButton} onPress={() => setEditModalVisible(false)}>
                <Text style={styles.modalCancelText}>Cancel</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.bg,
  },
  messageList: {
    padding: 16,
    paddingBottom: 24,
  },
  messageRow: {
    flexDirection: "row",
    marginBottom: 16,
    width: "100%",
  },
  userRow: {
    justifyContent: "flex-end",
  },
  assistantRow: {
    justifyContent: "flex-start",
  },
  bubble: {
    maxWidth: "85%",
    borderRadius: 16,
    padding: 12,
  },
  userBubble: {
    backgroundColor: COLORS.accent,
    borderBottomRightRadius: 4,
  },
  assistantBubble: {
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderBottomLeftRadius: 4,
  },
  messageText: {
    fontSize: 13,
    lineHeight: 18,
  },
  userText: {
    color: "#1a1611",
    fontWeight: "500",
  },
  assistantText: {
    color: COLORS.text,
  },
  inputContainer: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: COLORS.bg,
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
  },
  inputBar: {
    flexDirection: "row",
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 24,
    padding: 4,
    paddingLeft: 14,
    alignItems: "center",
  },
  textInput: {
    flex: 1,
    color: COLORS.text,
    fontSize: 13,
    paddingVertical: 6,
    maxHeight: 100,
  },
  sendButton: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: COLORS.accent,
    justifyContent: "center",
    alignItems: "center",
  },
  sendButtonArrow: {
    color: "#1a1611",
    fontWeight: "700",
    fontSize: 16,
    lineHeight: 18,
    textAlign: "center",
  },
  loadingIndicator: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 20,
    paddingVertical: 10,
    gap: 8,
  },
  loadingText: {
    color: COLORS.muted,
    fontFamily: "monospace",
    fontSize: 11,
  },
  
  // Proposed Schedule card styling
  scheduleCard: {
    width: "100%",
    backgroundColor: COLORS.surface,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 12,
    marginTop: 4,
  },
  scheduleHeaderRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 10,
    gap: 6,
  },
  headerDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: COLORS.accent,
  },
  scheduleHeader: {
    fontSize: 11,
    fontWeight: "700",
    color: COLORS.text,
    fontFamily: "monospace",
  },
  blockRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 6,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  blockBar: {
    width: 3,
    height: 14,
    borderRadius: 1.5,
    marginRight: 8,
  },
  blockTime: {
    fontSize: 11,
    color: COLORS.muted,
    fontFamily: "monospace",
    width: 48,
  },
  blockContent: {
    flex: 1,
  },
  blockTitle: {
    fontSize: 13,
    color: COLORS.text,
    fontWeight: "600",
  },
  blockNotes: {
    fontSize: 10,
    color: COLORS.muted,
    marginTop: 2,
  },
  blockDuration: {
    fontSize: 10,
    color: COLORS.muted,
    fontFamily: "monospace",
  },
  actionRow: {
    flexDirection: "row",
    gap: 10,
    marginTop: 16,
  },
  approveButton: {
    flex: 1,
    backgroundColor: COLORS.accent,
    borderRadius: 8,
    paddingVertical: 10,
    alignItems: "center",
  },
  actionButtonText: {
    color: "#0c0e14",
    fontWeight: "700",
    fontSize: 12,
  },
  rejectButton: {
    flex: 1,
    borderWidth: 1,
    borderColor: COLORS.accent,
    borderRadius: 8,
    paddingVertical: 10,
    alignItems: "center",
  },
  rejectButtonText: {
    color: COLORS.accent,
    fontWeight: "600",
    fontSize: 12,
  },

  // Modal styles
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(8, 9, 13, 0.85)",
    justifyContent: "flex-end",
  },
  modalContent: {
    backgroundColor: COLORS.surface,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 24,
    height: "80%",
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: COLORS.text,
    fontFamily: "monospace",
    marginBottom: 4,
  },
  modalSubtitle: {
    fontSize: 12,
    color: COLORS.muted,
    marginBottom: 20,
  },
  modalList: {
    paddingBottom: 24,
  },
  modalBlockContainer: {
    backgroundColor: COLORS.bg,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: COLORS.border,
    marginBottom: 8,
    overflow: "hidden",
  },
  modalBlockRow: {
    flexDirection: "row",
    alignItems: "center",
    borderLeftWidth: 4,
    paddingHorizontal: 16,
    paddingVertical: 14,
  },
  modalBlockTime: {
    fontSize: 13,
    color: COLORS.text,
    fontFamily: "monospace",
    width: 60,
    fontWeight: "600",
  },
  modalBlockType: {
    fontSize: 9,
    color: COLORS.muted,
    fontWeight: "700",
    marginTop: 2,
    letterSpacing: 0.5,
  },
  modalExpandIcon: {
    fontSize: 10,
    color: COLORS.muted,
    marginLeft: 8,
  },
  expandedControls: {
    backgroundColor: "rgba(30, 33, 48, 0.4)",
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
    padding: 16,
  },
  timeInputRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 12,
  },
  inputLabel: {
    color: COLORS.text,
    fontSize: 13,
    fontWeight: "600",
  },
  timeInput: {
    backgroundColor: COLORS.bg,
    color: COLORS.text,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 6,
    fontSize: 13,
    fontFamily: "monospace",
    width: 80,
    textAlign: "center",
  },
  removeBlockButton: {
    backgroundColor: "rgba(255, 79, 109, 0.1)",
    borderWidth: 1,
    borderColor: "rgba(255, 79, 109, 0.3)",
    borderRadius: 8,
    paddingVertical: 10,
    alignItems: "center",
  },
  removeBlockText: {
    color: COLORS.error,
    fontSize: 12,
    fontWeight: "600",
  },
  modalActionRow: {
    flexDirection: "row",
    gap: 12,
    marginTop: 16,
    paddingBottom: Platform.OS === "ios" ? 16 : 0,
  },
  modalSaveButton: {
    flex: 2,
    backgroundColor: COLORS.accent,
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: "center",
  },
  modalSaveText: {
    color: "#0c0e14",
    fontWeight: "700",
    fontSize: 14,
  },
  modalCancelButton: {
    flex: 1,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: "center",
  },
  modalCancelText: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: "600",
  },
  offlineBanner: {
    backgroundColor: "rgba(255, 79, 109, 0.15)",
    borderBottomWidth: 1,
    borderBottomColor: COLORS.error,
    paddingVertical: 10,
    alignItems: "center",
    justifyContent: "center",
    width: "100%",
  },
  offlineBannerText: {
    color: COLORS.error,
    fontSize: 12,
    fontWeight: "600",
    fontFamily: "monospace",
  },
});

import React, { useState, useEffect, useRef } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Modal,
  ActivityIndicator,
  Alert,
  Platform,
  KeyboardAvoidingView,
  Image,
  Animated,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { COLORS } from "../../constants/colors";
import { API_URL, authHeaders } from "../../services/apiHeaders";
import { firestoreService, authService } from "../../services/firebase";
import { SearchIcon, AddIcon, AIIcon, CameraIcon, MicIcon, WaveformIcon } from "../../components/icons";
import * as ImagePicker from "expo-image-picker";



// Conditionally import expo-av (may not be installed yet)
let Audio: any = null;
try {
  Audio = require("expo-av").Audio;
} catch (e) {
  // expo-av not installed — audio features will be disabled
}

export type NoteItem = {
  id: string;
  title: string;
  body: string;
  ai_summary?: string;
  subject?: string;
  tags?: string[];
  created_at?: string;
  type?: "text" | "photo" | "audio";
  image_url?: string;
  audio_url?: string;
  audio_duration_sec?: number;
  extracted_text?: string;
  transcript?: string;
  action_items?: string[];
  linked_syllabus?: {
    subject: string;
    unit_number: number;
    unit_title: string;
  };
};

export default function NotesScreen() {
  const [uid, setUid] = useState(authService.currentUser?.uid || "mock_user_123");
  const [notes, setNotes] = useState<NoteItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");

  // FAB menu
  const [fabOpen, setFabOpen] = useState(false);
  const fabAnim = useRef(new Animated.Value(0)).current;

  // Text note modal
  const [textModalVisible, setTextModalVisible] = useState(false);
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [tagsInput, setTagsInput] = useState("");
  const [saving, setSaving] = useState(false);

  // Photo note flow
  const [photoModalVisible, setPhotoModalVisible] = useState(false);
  const [capturedImageUri, setCapturedImageUri] = useState<string | null>(null);
  const [photoExtracting, setPhotoExtracting] = useState(false);
  const [photoExtractedData, setPhotoExtractedData] = useState<any>(null);
  const [photoEditText, setPhotoEditText] = useState("");
  const [photoEditSubject, setPhotoEditSubject] = useState("");
  const [photoEditTags, setPhotoEditTags] = useState("");
  const [photoSaving, setPhotoSaving] = useState(false);

  // Audio note flow
  const [audioModalVisible, setAudioModalVisible] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [audioProcessing, setAudioProcessing] = useState(false);
  const [audioExtractedData, setAudioExtractedData] = useState<any>(null);
  const [audioFileUri, setAudioFileUri] = useState<string | null>(null);
  const [audioSaving, setAudioSaving] = useState(false);
  const recordingRef = useRef<any>(null);
  const timerRef = useRef<any>(null);
  const pulseAnim = useRef(new Animated.Value(1)).current;

  // Audio playback
  const [playingNoteId, setPlayingNoteId] = useState<string | null>(null);

  // Expanded transcript
  const [expandedTranscript, setExpandedTranscript] = useState<string | null>(null);

  // Editing states
  const [editingNoteId, setEditingNoteId] = useState<string | null>(null);
  const [editingNoteType, setEditingNoteType] = useState<"text" | "photo" | "audio">("text");

  const MAX_RECORDING_SEC = 180; // 3 minutes

  useEffect(() => {
    const unsubAuth = authService.onAuthStateChanged((user: any) => {
      if (user) setUid(user.uid);
      else setUid("mock_user_123");
    });
    return unsubAuth;
  }, []);

  useEffect(() => {
    if (!uid) return;
    setLoading(true);
    const path = `notes/${uid}/items`;
    const unsub = firestoreService.onSnapshot(
      path,
      (list: any[]) => {
        const mapped = (list || []).map(item => ({
          id: item.id,
          title: item.title || "Untitled Note",
          body: item.body || item.extracted_text || item.transcript || "",
          ai_summary: item.ai_summary,
          subject: item.subject,
          tags: item.tags || [],
          created_at: item.created_at,
          type: item.type || "text",
          image_url: item.image_url,
          audio_url: item.audio_url,
          audio_duration_sec: item.audio_duration_sec,
          extracted_text: item.extracted_text,
          transcript: item.transcript,
          action_items: item.action_items,
          linked_syllabus: item.linked_syllabus,
        })).sort((a, b) => {
          const dateA = a.created_at || "";
          const dateB = b.created_at || "";
          return dateB.localeCompare(dateA);
        });
        setNotes(mapped);
        setLoading(false);
      },
      (err) => {
        console.error("Error fetching notes:", err);
        setLoading(false);
      }
    );
    return unsub;
  }, [uid]);

  // ─── FAB Menu Animation ──────────────────────────────────
  const toggleFab = () => {
    const toValue = fabOpen ? 0 : 1;
    Animated.spring(fabAnim, {
      toValue,
      friction: 6,
      tension: 80,
      useNativeDriver: true,
    }).start();
    setFabOpen(!fabOpen);
  };

  const closeFab = () => {
    Animated.spring(fabAnim, { toValue: 0, friction: 6, tension: 80, useNativeDriver: true }).start();
    setFabOpen(false);
  };

  // ─── TEXT NOTE ────────────────────────────────────────────
  const handleCreateTextNote = async () => {
    if (!body.trim()) {
      Alert.alert("Error", "Note body cannot be empty.");
      return;
    }
    setSaving(true);
    let aiSummary = "";
    let aiSubject = "General";
    let aiTags: string[] = [];
    let linkedSyllabus: any = null;

    try {
      const response = await fetch(`${API_URL}/api/notes/summarize`, {
        method: "POST",
        headers: await authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ note_content: body.trim(), user_id: uid }),
      });
      if (response.ok) {
        const data = await response.json();
        aiSummary = data.summary || "";
        aiSubject = data.subject || "General";
        aiTags = data.tags || [];
        linkedSyllabus = data.linked_syllabus || null;
      }
    } catch (err) {
      console.warn("AI Summarizer failed:", err);
    }

    const manualTags = tagsInput ? tagsInput.split(",").map(t => t.trim()).filter(Boolean) : [];
    const mergedTags = Array.from(new Set([...manualTags, ...aiTags]));
    const finalTitle = title.trim() || aiSubject || "Untitled Note";

    try {
      if (editingNoteId) {
        const updateData: any = {
          title: finalTitle,
          body: body.trim(),
          ai_summary: aiSummary || undefined,
          subject: aiSubject,
          tags: mergedTags,
        };
        if (linkedSyllabus) {
          updateData.linked_syllabus = linkedSyllabus;
        }
        if (editingNoteType === "photo") {
          updateData.extracted_text = body.trim();
        } else if (editingNoteType === "audio") {
          updateData.transcript = body.trim();
        }
        await firestoreService.update(`notes/${uid}/items/${editingNoteId}`, updateData);
      } else {
        const addData: any = {
          title: finalTitle,
          body: body.trim(),
          ai_summary: aiSummary || undefined,
          subject: aiSubject,
          tags: mergedTags,
          type: "text",
          created_at: new Date().toISOString(),
        };
        if (linkedSyllabus) {
          addData.linked_syllabus = linkedSyllabus;
        }
        await firestoreService.add(`notes/${uid}/items`, addData);
      }
      setTitle(""); setBody(""); setTagsInput(""); setEditingNoteId(null);
      setTextModalVisible(false);
    } catch (err: any) {
      Alert.alert("Error", `Failed to save note: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  // ─── PHOTO NOTE ───────────────────────────────────────────
  const handleCapturePhoto = async (fromCamera: boolean) => {
    closeFab();
    try {
      const result = fromCamera
        ? await ImagePicker.launchCameraAsync({ mediaTypes: ['images'], quality: 0.6, base64: false })
        : await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 0.6, base64: false });

      if (result.canceled || !result.assets?.[0]) return;

      const asset = result.assets[0];
      setCapturedImageUri(asset.uri);
      setPhotoExtractedData(null);
      setPhotoEditText("");
      setPhotoEditSubject("");
      setPhotoEditTags("");
      setPhotoModalVisible(true);
      setPhotoExtracting(true);

      // Upload to backend for Gemini Vision extraction
      const formData = new FormData();
      formData.append("user_id", uid);
      formData.append("file", {
        uri: asset.uri,
        name: asset.fileName || "photo.jpg",
        type: "image/jpeg",
      } as any);

      const response = await fetch(`${API_URL}/api/upload/photo-note`, {
        method: "POST",
        headers: await authHeaders(),
        body: formData,
      });

      if (response.ok) {
        const json = await response.json();
        const data = json.data || json;
        setPhotoExtractedData(data);
        setPhotoEditText(data.extracted_text || "");
        setPhotoEditSubject(data.subject || "General");
        setPhotoEditTags((data.tags || []).join(", "));
      } else {
        Alert.alert("Extraction Failed", "Could not extract text from the image.");
      }
    } catch (err: any) {
      Alert.alert("Error", err.message || "Failed to capture photo.");
    } finally {
      setPhotoExtracting(false);
    }
  };

  const handleSavePhotoNote = async () => {
    if (!photoEditText.trim() && !photoExtractedData?.summary) {
      Alert.alert("Error", "No content to save.");
      return;
    }
    setPhotoSaving(true);
    try {
      const tags = photoEditTags ? photoEditTags.split(",").map(t => t.trim()).filter(Boolean) : [];
      const addData: any = {
        title: photoExtractedData?.summary?.substring(0, 60) || "Photo Note",
        body: photoEditText.trim(),
        extracted_text: photoEditText.trim(),
        ai_summary: photoExtractedData?.summary || "",
        subject: photoEditSubject || "General",
        tags,
        type: "photo",
        image_url: capturedImageUri, // Local URI for now
        created_at: new Date().toISOString(),
      };
      if (photoExtractedData?.linked_syllabus) {
        addData.linked_syllabus = photoExtractedData.linked_syllabus;
      }
      await firestoreService.add(`notes/${uid}/items`, addData);
      setPhotoModalVisible(false);
      setCapturedImageUri(null);
      setPhotoExtractedData(null);
    } catch (err: any) {
      Alert.alert("Error", `Failed to save photo note: ${err.message}`);
    } finally {
      setPhotoSaving(false);
    }
  };

  // ─── AUDIO NOTE ───────────────────────────────────────────
  const startRecording = async () => {
    if (!Audio) {
      Alert.alert("Not Available", "Audio recording requires expo-av. Please run an EAS build with expo-av installed.");
      return;
    }
    try {
      const perm = await Audio.requestPermissionsAsync();
      if (!perm.granted) {
        Alert.alert("Permission Required", "Microphone permission is needed to record audio.");
        return;
      }
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });
      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      recordingRef.current = recording;
      setIsRecording(true);
      setRecordingDuration(0);

      // Start pulse animation
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 1.3, duration: 600, useNativeDriver: true }),
          Animated.timing(pulseAnim, { toValue: 1, duration: 600, useNativeDriver: true }),
        ])
      ).start();

      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingDuration(prev => {
          if (prev + 1 >= MAX_RECORDING_SEC) {
            stopRecording(true);
            return MAX_RECORDING_SEC;
          }
          return prev + 1;
        });
      }, 1000);
    } catch (err: any) {
      Alert.alert("Error", `Failed to start recording: ${err.message}`);
    }
  };

  const stopRecording = async (autoStopped = false) => {
    if (timerRef.current) clearInterval(timerRef.current);
    pulseAnim.stopAnimation();
    pulseAnim.setValue(1);

    if (!recordingRef.current) return;
    try {
      await recordingRef.current.stopAndUnloadAsync();
      const uri = recordingRef.current.getURI();
      recordingRef.current = null;
      setIsRecording(false);

      if (autoStopped) {
        Alert.alert("Recording Capped", "Recording capped at 3 minutes for voice notes.\nFor longer recordings, use Lecture Mode (coming soon).");
      }

      if (!uri) return;
      setAudioFileUri(uri);
      setAudioProcessing(true);
      setAudioExtractedData(null);

      // Upload to backend
      const formData = new FormData();
      formData.append("user_id", uid);
      formData.append("file", {
        uri,
        name: "voice_note.m4a",
        type: "audio/mp4",
      } as any);

      const response = await fetch(`${API_URL}/api/upload/audio-note`, {
        method: "POST",
        headers: await authHeaders(),
        body: formData,
      });

      if (response.ok) {
        const json = await response.json();
        const data = json.data || json;
        setAudioExtractedData(data);
      } else {
        const errText = await response.text();
        Alert.alert("Processing Failed", errText || "Could not process audio.");
      }
    } catch (err: any) {
      Alert.alert("Error", `Failed to process recording: ${err.message}`);
    } finally {
      setAudioProcessing(false);
    }
  };

  const handleSaveAudioNote = async () => {
    if (!audioExtractedData) return;
    setAudioSaving(true);
    try {
      const data = audioExtractedData;
      const addData: any = {
        title: data.summary?.substring(0, 60) || "Voice Note",
        body: data.transcript || "",
        transcript: data.transcript || "",
        ai_summary: data.summary || "",
        subject: data.subject || "General",
        tags: data.tags || [],
        type: "audio",
        audio_url: audioFileUri,
        audio_duration_sec: recordingDuration,
        action_items: data.action_items || [],
        created_at: new Date().toISOString(),
      };
      if (data.linked_syllabus) {
        addData.linked_syllabus = data.linked_syllabus;
      }
      await firestoreService.add(`notes/${uid}/items`, addData);
      setAudioModalVisible(false);
      setAudioExtractedData(null);
      setAudioFileUri(null);
      setRecordingDuration(0);
    } catch (err: any) {
      Alert.alert("Error", `Failed to save voice note: ${err.message}`);
    } finally {
      setAudioSaving(false);
    }
  };

  // ─── AUDIO PLAYBACK ───────────────────────────────────────
  const playAudio = async (noteId: string, audioUri: string) => {
    if (!Audio) return;
    try {
      if (playingNoteId === noteId) {
        setPlayingNoteId(null);
        return;
      }
      const { sound } = await Audio.Sound.createAsync({ uri: audioUri });
      setPlayingNoteId(noteId);
      sound.setOnPlaybackStatusUpdate((status: any) => {
        if (status.didJustFinish) {
          setPlayingNoteId(null);
          sound.unloadAsync();
        }
      });
      await sound.playAsync();
    } catch (err) {
      console.warn("Playback error:", err);
    }
  };

  // ─── DELETE ───────────────────────────────────────────────
  const handleDeleteNote = (noteId: string) => {
    Alert.alert("Delete Note", "Are you sure you want to delete this note?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Delete",
        style: "destructive",
        onPress: async () => {
          try {
            await firestoreService.delete(`notes/${uid}/items/${noteId}`);
          } catch (err: any) {
            Alert.alert("Error", `Failed to delete note: ${err.message}`);
          }
        },
      },
    ]);
  };

  // ─── SEARCH FILTER ────────────────────────────────────────
  const filteredNotes = notes.filter(n => {
    const query = searchQuery.toLowerCase();
    const titleMatch = n.title.toLowerCase().includes(query);
    const bodyMatch = n.body.toLowerCase().includes(query);
    const tagsMatch = n.tags?.some(tag => tag.toLowerCase().includes(query));
    return titleMatch || bodyMatch || tagsMatch;
  });

  const formatDuration = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  // ─── NOTE CARD RENDERER ──────────────────────────────────
  const handleEditNote = (n: NoteItem) => {
    setEditingNoteId(n.id);
    setEditingNoteType(n.type || "text");
    setTitle(n.title);
    setBody(n.body);
    setTagsInput(n.tags ? n.tags.join(", ") : "");
    setTextModalVisible(true);
  };

  const renderNoteCard = (n: NoteItem) => {
    const isAudio = n.type === "audio";
    const isPhoto = n.type === "photo";

    return (
      <TouchableOpacity
        key={n.id}
        style={styles.noteCard}
        onPress={() => handleEditNote(n)}
        onLongPress={() => handleDeleteNote(n.id)}
      >
        <View style={styles.cardTopRow}>
          {/* Photo thumbnail */}
          {isPhoto && n.image_url && (
            <Image source={{ uri: n.image_url }} style={styles.photoThumb} />
          )}

          {/* Audio play button */}
          {isAudio && n.audio_url && (
            <TouchableOpacity
              style={[styles.playBtn, playingNoteId === n.id && styles.playBtnActive]}
              onPress={() => playAudio(n.id, n.audio_url!)}
            >
              <Text style={styles.playBtnText}>{playingNoteId === n.id ? "⏹" : "▶"}</Text>
            </TouchableOpacity>
          )}

          <View style={styles.cardContent}>
            {/* Type badge + Tags row */}
            <View style={styles.noteTags}>
              {isPhoto && (
                <View style={[styles.noteTag, styles.typeBadge]}>
                  <Text style={styles.typeBadgeText}>📷 Photo</Text>
                </View>
              )}
              {isAudio && (
                <View style={[styles.noteTag, styles.typeBadge]}>
                  <Text style={styles.typeBadgeText}>🎙️ {n.audio_duration_sec ? formatDuration(n.audio_duration_sec) : "Voice"}</Text>
                </View>
              )}
              {n.linked_syllabus && (
                <View style={[styles.noteTag, styles.syllabusBadge]}>
                  <Text style={styles.syllabusBadgeText}>📚 {n.linked_syllabus.subject} · U{n.linked_syllabus.unit_number}</Text>
                </View>
              )}
              {n.tags && n.tags.map((tag, idx) => (
                <View key={idx} style={styles.noteTag}>
                  <Text style={styles.noteTagText}>{tag}</Text>
                </View>
              ))}
            </View>

            <Text style={styles.noteTitle}>{n.title}</Text>
            <Text style={styles.noteBody} numberOfLines={isAudio ? 2 : 3}>
              {n.body}
            </Text>
          </View>
        </View>

        {/* Audio transcript toggle */}
        {isAudio && n.transcript && (
          <TouchableOpacity
            style={styles.transcriptToggle}
            onPress={() => setExpandedTranscript(expandedTranscript === n.id ? null : n.id)}
          >
            <Text style={styles.transcriptToggleText}>
              {expandedTranscript === n.id ? "Hide transcript ▲" : "Show transcript ▼"}
            </Text>
          </TouchableOpacity>
        )}
        {isAudio && expandedTranscript === n.id && n.transcript && (
          <Text style={styles.transcriptText}>{n.transcript}</Text>
        )}

        {/* Action items from audio */}
        {isAudio && n.action_items && n.action_items.length > 0 && (
          <View style={styles.actionItemsBox}>
            <Text style={styles.actionItemsTitle}>📋 Auto-created tasks:</Text>
            {n.action_items.map((item, idx) => (
              <Text key={idx} style={styles.actionItemText}>• {item}</Text>
            ))}
          </View>
        )}

        {/* AI Summary */}
        {n.ai_summary ? (
          <View style={styles.noteAi}>
            <View style={styles.aiIconWrapper}>
              <AIIcon size={12} active={true} />
            </View>
            <Text style={styles.noteAiText}>
              <Text style={{ fontWeight: "700" }}>AI Summary: </Text>
              {n.ai_summary}
            </Text>
          </View>
        ) : null}
      </TouchableOpacity>
    );
  };

  // ─── MAIN RENDER ──────────────────────────────────────────
  return (
    <SafeAreaView style={styles.container} edges={["left", "right"]}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Notes</Text>
      </View>

      {/* SEARCH BAR */}
      <View style={styles.searchContainer}>
        <View style={styles.searchBar}>
          <SearchIcon size={16} />
          <TextInput
            style={styles.searchText}
            value={searchQuery}
            onChangeText={setSearchQuery}
            placeholder="Search notes..."
            placeholderTextColor={COLORS.textDim}
          />
        </View>
      </View>

      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="small" color={COLORS.accent} />
        </View>
      ) : (
        <ScrollView contentContainerStyle={styles.scrollContainer}>
          {filteredNotes.map(renderNoteCard)}
          {filteredNotes.length === 0 && (
            <View style={styles.emptyContainer}>
              <Text style={styles.emptyText}>No notes found</Text>
            </View>
          )}
        </ScrollView>
      )}

      {/* ─── FAB MENU ──────────────────────────────────────── */}
      {fabOpen && (
        <TouchableOpacity style={styles.fabOverlay} activeOpacity={1} onPress={closeFab} />
      )}

      {/* Voice Note option */}
      <Animated.View style={[styles.fabOption, {
        opacity: fabAnim,
        transform: [{ translateY: fabAnim.interpolate({ inputRange: [0, 1], outputRange: [0, -170] }) }],
      }]} pointerEvents={fabOpen ? "auto" : "none"}>
        <TouchableOpacity style={styles.fabOptionBtn} onPress={() => {
          closeFab();
          if (!Audio) {
            Alert.alert("Not Available", "Audio recording requires expo-av.\nInstall with: npx expo install expo-av\nThen run an EAS development build.");
            return;
          }
          setAudioExtractedData(null);
          setAudioFileUri(null);
          setRecordingDuration(0);
          setIsRecording(false);
          setAudioModalVisible(true);
        }}>
          <MicIcon size={18} active />
          <Text style={styles.fabOptionLabel}>Voice Note</Text>
        </TouchableOpacity>
      </Animated.View>

      {/* Photo Note option */}
      <Animated.View style={[styles.fabOption, {
        opacity: fabAnim,
        transform: [{ translateY: fabAnim.interpolate({ inputRange: [0, 1], outputRange: [0, -120] }) }],
      }]} pointerEvents={fabOpen ? "auto" : "none"}>
        <TouchableOpacity style={styles.fabOptionBtn} onPress={() => handleCapturePhoto(true)}>
          <CameraIcon size={18} active />
          <Text style={styles.fabOptionLabel}>Photo Note</Text>
        </TouchableOpacity>
      </Animated.View>

      {/* Text Note option */}
      <Animated.View style={[styles.fabOption, {
        opacity: fabAnim,
        transform: [{ translateY: fabAnim.interpolate({ inputRange: [0, 1], outputRange: [0, -70] }) }],
      }]} pointerEvents={fabOpen ? "auto" : "none"}>
        <TouchableOpacity style={styles.fabOptionBtn} onPress={() => {
          closeFab();
          setTitle(""); setBody(""); setTagsInput("");
          setEditingNoteId(null);
          setTextModalVisible(true);
        }}>
          <AddIcon size={18} active />
          <Text style={styles.fabOptionLabel}>Text Note</Text>
        </TouchableOpacity>
      </Animated.View>

      {/* Main FAB */}
      <TouchableOpacity style={styles.fab} onPress={toggleFab}>
        <Animated.View style={{ transform: [{ rotate: fabAnim.interpolate({ inputRange: [0, 1], outputRange: ["0deg", "45deg"] }) }] }}>
          <AddIcon size={22} color="#15161c" />
        </Animated.View>
      </TouchableOpacity>

      {/* ─── TEXT NOTE MODAL ──────────────────────────────── */}
      <Modal visible={textModalVisible} animationType="slide" transparent={true}>
        <View style={styles.modalOverlay}>
          <KeyboardAvoidingView
            behavior={Platform.OS === "ios" ? "padding" : "height"}
            style={styles.modalContent}
          >
            <Text style={styles.modalTitle}>
              {editingNoteId ? `📝 Edit Note (${editingNoteType})` : "📝 Text Note"}
            </Text>
            <ScrollView style={{ width: "100%" }}>
              <View style={styles.formGroup}>
                <Text style={styles.formLabel}>Title (Optional)</Text>
                <TextInput style={styles.input} value={title} onChangeText={setTitle}
                  placeholder="e.g. Normalization Notes" placeholderTextColor={COLORS.textDim} />
              </View>
              <View style={styles.formGroup}>
                <Text style={styles.formLabel}>Tags (Comma-separated)</Text>
                <TextInput style={styles.input} value={tagsInput} onChangeText={setTagsInput}
                  placeholder="e.g. DBMS, CIE-1" placeholderTextColor={COLORS.textDim} />
              </View>
              <View style={styles.formGroup}>
                <Text style={styles.formLabel}>Note Content</Text>
                <TextInput style={[styles.input, styles.textArea]} value={body} onChangeText={setBody}
                  placeholder="Type notes content here..." placeholderTextColor={COLORS.textDim}
                  multiline numberOfLines={6} />
              </View>
            </ScrollView>
            <View style={styles.modalActionRow}>
              <TouchableOpacity style={styles.saveBtn} onPress={handleCreateTextNote} disabled={saving}>
                {saving ? <ActivityIndicator size="small" color="#15161c" /> : <Text style={styles.saveBtnText}>{editingNoteId ? "Save Changes" : "Save Note"}</Text>}
              </TouchableOpacity>
              <TouchableOpacity style={styles.cancelBtn} onPress={() => setTextModalVisible(false)}>
                <Text style={styles.cancelBtnText}>Cancel</Text>
              </TouchableOpacity>
            </View>
          </KeyboardAvoidingView>
        </View>
      </Modal>

      {/* ─── PHOTO NOTE MODAL (Confirmation) ─────────────── */}
      <Modal visible={photoModalVisible} animationType="slide" transparent={true}>
        <View style={styles.modalOverlay}>
          <KeyboardAvoidingView
            behavior={Platform.OS === "ios" ? "padding" : "height"}
            style={styles.modalContent}
          >
            <Text style={styles.modalTitle}>📷 Photo Note</Text>
            <ScrollView style={{ width: "100%" }} keyboardShouldPersistTaps="handled">
              {/* Image preview */}
              {capturedImageUri && (
                <Image source={{ uri: capturedImageUri }} style={styles.photoPreview} resizeMode="contain" />
              )}

              {photoExtracting ? (
                <View style={styles.extractingBox}>
                  <ActivityIndicator size="small" color={COLORS.accent} />
                  <Text style={styles.extractingText}>Reading image...</Text>
                </View>
              ) : photoExtractedData ? (
                <>
                  {/* AI Summary */}
                  {photoExtractedData.summary && (
                    <View style={styles.noteAi}>
                      <AIIcon size={12} active={true} />
                      <Text style={styles.noteAiText}>
                        <Text style={{ fontWeight: "700" }}>Summary: </Text>
                        {photoExtractedData.summary}
                      </Text>
                    </View>
                  )}

                  {/* Key points */}
                  {photoExtractedData.key_points?.length > 0 && (
                    <View style={styles.keyPointsBox}>
                      <Text style={styles.keyPointsTitle}>Key Points:</Text>
                      {photoExtractedData.key_points.map((p: string, i: number) => (
                        <Text key={i} style={styles.keyPointText}>• {p}</Text>
                      ))}
                    </View>
                  )}

                  {/* Editable extracted text */}
                  <View style={styles.formGroup}>
                    <Text style={styles.formLabel}>Extracted Text (editable)</Text>
                    <TextInput style={[styles.input, styles.textArea]} value={photoEditText}
                      onChangeText={setPhotoEditText} multiline numberOfLines={6} />
                  </View>
                  <View style={styles.formGroup}>
                    <Text style={styles.formLabel}>Subject</Text>
                    <TextInput style={styles.input} value={photoEditSubject}
                      onChangeText={setPhotoEditSubject} />
                  </View>
                  <View style={styles.formGroup}>
                    <Text style={styles.formLabel}>Tags (comma-separated)</Text>
                    <TextInput style={styles.input} value={photoEditTags}
                      onChangeText={setPhotoEditTags} />
                  </View>
                </>
              ) : null}
            </ScrollView>

            <View style={styles.modalActionRow}>
              <TouchableOpacity style={styles.saveBtn} onPress={handleSavePhotoNote}
                disabled={photoSaving || photoExtracting}>
                {photoSaving ? <ActivityIndicator size="small" color="#15161c" /> : <Text style={styles.saveBtnText}>Save</Text>}
              </TouchableOpacity>
              <TouchableOpacity style={styles.cancelBtn} onPress={() => setPhotoModalVisible(false)}>
                <Text style={styles.cancelBtnText}>Cancel</Text>
              </TouchableOpacity>
            </View>
          </KeyboardAvoidingView>
        </View>
      </Modal>

      {/* ─── AUDIO NOTE MODAL ────────────────────────────── */}
      <Modal visible={audioModalVisible} animationType="slide" transparent={true}>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>🎙️ Voice Note</Text>

            {!audioExtractedData && !audioProcessing ? (
              /* Recording state */
              <View style={styles.recordingContainer}>
                {isRecording ? (
                  <>
                    <Animated.View style={[styles.recordDot, { transform: [{ scale: pulseAnim }] }]} />
                    <Text style={styles.recordTimer}>{formatDuration(recordingDuration)}</Text>
                    <Text style={styles.recordLimit}>Max: {formatDuration(MAX_RECORDING_SEC)}</Text>
                    <TouchableOpacity style={styles.stopBtn} onPress={() => stopRecording(false)}>
                      <Text style={styles.stopBtnText}>⏹ Stop</Text>
                    </TouchableOpacity>
                  </>
                ) : (
                  <>
                    <Text style={styles.recordPrompt}>Tap to start recording</Text>
                    <Text style={styles.recordLimit}>Max: 3 minutes</Text>
                    <TouchableOpacity style={styles.recordBtn} onPress={startRecording}>
                      <MicIcon size={28} active />
                    </TouchableOpacity>
                  </>
                )}
              </View>
            ) : audioProcessing ? (
              <View style={styles.extractingBox}>
                <ActivityIndicator size="small" color={COLORS.accent} />
                <Text style={styles.extractingText}>Processing audio...</Text>
                <Text style={[styles.extractingText, { fontSize: 10, marginTop: 4 }]}>This may take a moment</Text>
              </View>
            ) : audioExtractedData ? (
              <ScrollView style={{ width: "100%" }}>
                {/* Summary */}
                {audioExtractedData.summary && (
                  <View style={styles.noteAi}>
                    <AIIcon size={12} active={true} />
                    <Text style={styles.noteAiText}>
                      <Text style={{ fontWeight: "700" }}>Summary: </Text>
                      {audioExtractedData.summary}
                    </Text>
                  </View>
                )}

                {/* Transcript preview */}
                <View style={styles.formGroup}>
                  <Text style={styles.formLabel}>Transcript</Text>
                  <Text style={styles.transcriptPreview} numberOfLines={8}>
                    {audioExtractedData.transcript || "No transcript available"}
                  </Text>
                </View>

                {/* Action items */}
                {audioExtractedData.action_items?.length > 0 && (
                  <View style={styles.actionItemsBox}>
                    <Text style={styles.actionItemsTitle}>📋 Tasks auto-created:</Text>
                    {audioExtractedData.action_items.map((item: string, idx: number) => (
                      <Text key={idx} style={styles.actionItemText}>✓ {item}</Text>
                    ))}
                  </View>
                )}

                {/* Tags */}
                {audioExtractedData.tags?.length > 0 && (
                  <View style={styles.noteTags}>
                    {audioExtractedData.tags.map((tag: string, idx: number) => (
                      <View key={idx} style={styles.noteTag}>
                        <Text style={styles.noteTagText}>{tag}</Text>
                      </View>
                    ))}
                  </View>
                )}
              </ScrollView>
            ) : null}

            <View style={styles.modalActionRow}>
              {audioExtractedData && (
                <TouchableOpacity style={styles.saveBtn} onPress={handleSaveAudioNote} disabled={audioSaving}>
                  {audioSaving ? <ActivityIndicator size="small" color="#15161c" /> : <Text style={styles.saveBtnText}>Save</Text>}
                </TouchableOpacity>
              )}
              <TouchableOpacity style={[styles.cancelBtn, !audioExtractedData && { flex: 1 }]}
                onPress={() => {
                  if (isRecording) stopRecording(false);
                  setAudioModalVisible(false);
                }}>
                <Text style={styles.cancelBtnText}>{isRecording ? "Cancel" : "Close"}</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.bg },
  header: {
    flexDirection: "row", justifyContent: "space-between", alignItems: "center",
    paddingHorizontal: 20, paddingTop: 16, paddingBottom: 8,
  },
  headerTitle: { fontSize: 24, fontWeight: "800", color: COLORS.text },
  searchContainer: { paddingHorizontal: 20, marginBottom: 16 },
  searchBar: {
    flexDirection: "row", alignItems: "center", backgroundColor: COLORS.surface,
    borderWidth: 1, borderColor: COLORS.border, borderRadius: 24,
    paddingHorizontal: 16, paddingVertical: 10, gap: 10,
  },
  searchText: { flex: 1, color: COLORS.text, fontSize: 14, padding: 0 },
  scrollContainer: { paddingHorizontal: 20, paddingBottom: 100 },
  loadingContainer: { flex: 1, justifyContent: "center", alignItems: "center" },
  emptyContainer: { paddingVertical: 48, alignItems: "center" },
  emptyText: { color: COLORS.textDim, fontSize: 14 },

  // ─── Note card ────────────────────────────────────────────
  noteCard: {
    backgroundColor: COLORS.surface, borderWidth: 1, borderColor: COLORS.border,
    borderRadius: 20, padding: 16, marginBottom: 12,
  },
  cardTopRow: { flexDirection: "row", gap: 12 },
  cardContent: { flex: 1 },
  photoThumb: {
    width: 60, height: 60, borderRadius: 10, backgroundColor: COLORS.panel,
  },
  playBtn: {
    width: 44, height: 44, borderRadius: 22, backgroundColor: COLORS.panel,
    justifyContent: "center", alignItems: "center", borderWidth: 1, borderColor: COLORS.border,
  },
  playBtnActive: { backgroundColor: COLORS.accentBg, borderColor: COLORS.accent },
  playBtnText: { fontSize: 16 },
  noteTags: { flexDirection: "row", flexWrap: "wrap", gap: 6, marginBottom: 8 },
  noteTag: { backgroundColor: COLORS.panel, borderRadius: 12, paddingHorizontal: 8, paddingVertical: 3 },
  noteTagText: { fontSize: 9, fontWeight: "600", color: COLORS.textMuted },
  typeBadge: { backgroundColor: COLORS.accentBg, borderWidth: 1, borderColor: COLORS.accentLine },
  typeBadgeText: { fontSize: 9, fontWeight: "700", color: COLORS.accent },
  noteTitle: { fontSize: 15, fontWeight: "700", color: COLORS.text, marginBottom: 6 },
  noteBody: { fontSize: 12, color: COLORS.textMuted, lineHeight: 18, marginBottom: 10 },

  // Transcript
  transcriptToggle: { paddingVertical: 6 },
  transcriptToggleText: { fontSize: 10, color: COLORS.accent, fontWeight: "600" },
  transcriptText: {
    fontSize: 11, color: COLORS.textMuted, lineHeight: 16, marginBottom: 10,
    backgroundColor: COLORS.bg, padding: 10, borderRadius: 8,
  },

  // Action items
  actionItemsBox: {
    backgroundColor: "rgba(110,231,168,0.08)", borderRadius: 10,
    padding: 10, marginBottom: 10, borderWidth: 1, borderColor: "rgba(110,231,168,0.2)",
  },
  actionItemsTitle: { fontSize: 10, fontWeight: "700", color: COLORS.green, marginBottom: 4 },
  actionItemText: { fontSize: 11, color: COLORS.green, lineHeight: 16 },

  // AI summary
  noteAi: {
    flexDirection: "row", alignItems: "flex-start", backgroundColor: COLORS.lavenderBg,
    borderWidth: 1, borderColor: "rgba(179,161,255,0.2)", borderRadius: 12,
    padding: 10, gap: 8, marginBottom: 10,
  },
  aiIconWrapper: { marginTop: 2 },
  noteAiText: { flex: 1, fontSize: 11, color: COLORS.lavender, lineHeight: 16 },

  // ─── FAB ──────────────────────────────────────────────────
  fabOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(0,0,0,0.3)",
  },
  fab: {
    position: "absolute", right: 20, bottom: 20, width: 52, height: 52, borderRadius: 26,
    backgroundColor: COLORS.accent, justifyContent: "center", alignItems: "center",
    shadowColor: "#000", shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3, shadowRadius: 6, elevation: 5, zIndex: 20,
  },
  fabOption: {
    position: "absolute", right: 20, bottom: 20, zIndex: 15,
  },
  fabOptionBtn: {
    flexDirection: "row", alignItems: "center", gap: 10,
    backgroundColor: COLORS.surface, borderWidth: 1, borderColor: COLORS.border,
    borderRadius: 24, paddingHorizontal: 16, paddingVertical: 10,
    shadowColor: "#000", shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2, shadowRadius: 4, elevation: 3,
  },
  fabOptionLabel: {
    color: COLORS.text, fontSize: 13, fontWeight: "600",
  },

  // ─── Modals ───────────────────────────────────────────────
  modalOverlay: { flex: 1, backgroundColor: "rgba(0,0,0,0.7)", justifyContent: "flex-end" },
  modalContent: {
    backgroundColor: COLORS.surface, borderTopLeftRadius: 24, borderTopRightRadius: 24,
    borderWidth: 1, borderColor: COLORS.border, padding: 24,
    maxHeight: "90%", alignItems: "center",
  },
  modalTitle: { fontSize: 18, fontWeight: "800", color: COLORS.text, marginBottom: 16 },
  formGroup: { width: "100%", marginBottom: 16 },
  formLabel: {
    fontSize: 11, color: COLORS.textDim, fontWeight: "700",
    textTransform: "uppercase", marginBottom: 6, fontFamily: "monospace",
  },
  input: {
    backgroundColor: COLORS.bg, color: COLORS.text, borderWidth: 1,
    borderColor: COLORS.border, borderRadius: 12, padding: 12, fontSize: 14,
  },
  textArea: { textAlignVertical: "top", minHeight: 120 },
  modalActionRow: { flexDirection: "row", gap: 12, marginTop: 16, width: "100%" },
  saveBtn: {
    flex: 2, backgroundColor: COLORS.accent, borderRadius: 12,
    paddingVertical: 14, alignItems: "center",
  },
  saveBtnText: { color: "#15161c", fontWeight: "700", fontSize: 14 },
  cancelBtn: {
    flex: 1, borderWidth: 1, borderColor: COLORS.border, borderRadius: 12,
    paddingVertical: 14, alignItems: "center",
  },
  cancelBtnText: { color: COLORS.text, fontSize: 14, fontWeight: "600" },

  // ─── Photo modal ──────────────────────────────────────────
  photoPreview: {
    width: "100%", height: 200, borderRadius: 12, marginBottom: 16,
    backgroundColor: COLORS.bg,
  },
  extractingBox: {
    paddingVertical: 32, alignItems: "center", gap: 10,
  },
  extractingText: { color: COLORS.textMuted, fontSize: 13 },
  keyPointsBox: {
    backgroundColor: COLORS.bg, borderRadius: 10, padding: 12, marginBottom: 16,
    borderWidth: 1, borderColor: COLORS.border,
  },
  keyPointsTitle: { fontSize: 11, fontWeight: "700", color: COLORS.accent, marginBottom: 6 },
  keyPointText: { fontSize: 11, color: COLORS.textMuted, lineHeight: 16 },

  // ─── Audio modal ──────────────────────────────────────────
  recordingContainer: {
    paddingVertical: 40, alignItems: "center", gap: 16, width: "100%",
  },
  recordDot: {
    width: 20, height: 20, borderRadius: 10, backgroundColor: COLORS.red,
  },
  recordTimer: {
    fontSize: 40, fontWeight: "800", color: COLORS.text, fontFamily: "monospace",
  },
  recordLimit: { fontSize: 11, color: COLORS.textDim },
  recordPrompt: { fontSize: 15, fontWeight: "600", color: COLORS.textMuted },
  recordBtn: {
    width: 64, height: 64, borderRadius: 32, backgroundColor: COLORS.accentBg,
    borderWidth: 2, borderColor: COLORS.accent,
    justifyContent: "center", alignItems: "center",
  },
  stopBtn: {
    paddingHorizontal: 24, paddingVertical: 12, borderRadius: 20,
    backgroundColor: "rgba(255,138,155,0.15)", borderWidth: 1, borderColor: COLORS.red,
  },
  stopBtnText: { color: COLORS.red, fontSize: 14, fontWeight: "700" },
  transcriptPreview: {
    fontSize: 12, color: COLORS.textMuted, lineHeight: 18,
    backgroundColor: COLORS.bg, padding: 12, borderRadius: 10,
    borderWidth: 1, borderColor: COLORS.border,
  },
  syllabusBadge: {
    backgroundColor: COLORS.lavenderBg,
    borderWidth: 1,
    borderColor: "rgba(179,161,255,0.2)",
  },
  syllabusBadgeText: {
    fontSize: 9,
    fontWeight: "700",
    color: COLORS.lavender,
  },
});

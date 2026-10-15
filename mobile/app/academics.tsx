import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  ActivityIndicator,
  Alert,
  Modal,
  Platform,
  KeyboardAvoidingView
} from "react-native";
import { useRouter } from "expo-router";
import * as ImagePicker from "expo-image-picker";
import * as DocumentPicker from "expo-document-picker";
import * as FileSystem from "expo-file-system/legacy";
import { COLORS } from "../constants/colors";
import { firestoreService, authService } from "../services/firebase";
import { API_URL, authHeaders } from "../services/apiHeaders";

type AcademicEvent = {
  id?: string;
  title: string;
  type: "cie" | "see" | "assignment" | "lab_record" | "viva";
  date: string;
  description?: string;
  confidence?: "scraped" | "user_confirmed" | "manual";
};

export default function AcademicsScreen() {
  const router = useRouter();
  const [uid, setUid] = useState(authService.currentUser?.uid || "mock_user_123");

  const [events, setEvents] = useState<AcademicEvent[]>([]);
  const [loadingEvents, setLoadingEvents] = useState(true);

  // Form states for manual academic event addition
  const [showAddForm, setShowAddForm] = useState(false);
  const [title, setTitle] = useState("");
  const [type, setType] = useState<"cie" | "see" | "assignment" | "lab_record" | "viva">("cie");
  const [date, setDate] = useState("");
  const [description, setDescription] = useState("");
  const [savingEvent, setSavingEvent] = useState(false);

  // Upload states
  const [uploading, setUploading] = useState<"timetable" | "syllabus" | null>(null);
  
  // Subject name for syllabus upload
  const [syllabusSubject, setSyllabusSubject] = useState("");
  const [showSubjectPrompt, setShowSubjectPrompt] = useState(false);
  const [syllabusFile, setSyllabusFile] = useState<any>(null);

  // Confirmation Modals states
  const [showTimetableModal, setShowTimetableModal] = useState(false);
  const [rawTimetableJson, setRawTimetableJson] = useState("");
  const [showSyllabusModal, setShowSyllabusModal] = useState(false);
  const [rawSyllabusJson, setRawSyllabusJson] = useState("");
  const [timetableData, setTimetableData] = useState<Record<string, Array<{time: string, subject: string, room: string, professor: string}>>>({});
  const [isEditingTimetable, setIsEditingTimetable] = useState(false);

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

  useEffect(() => {
    if (!uid) {
      setEvents([]);
      setLoadingEvents(false);
      return;
    }
    setLoadingEvents(true);
    const docPath = `academic_events/${uid}/items`;
    const unsub = firestoreService.onSnapshot(
      docPath,
      (list: any[]) => {
        const sorted = (list || []).sort((a, b) => a.date.localeCompare(b.date));
        setEvents(sorted);
        setLoadingEvents(false);
      },
      (err: any) => {
        console.error("Error fetching academic events:", err);
        setLoadingEvents(false);
      }
    );
    return unsub;
  }, [uid]);

  const handleAddEvent = async () => {
    if (!title.trim() || !date.trim()) {
      Alert.alert("Error", "Please fill in the title and date (YYYY-MM-DD) fields.");
      return;
    }

    if (!/^\d{4}-\d{2}-\d{2}$/.test(date.trim())) {
      Alert.alert("Error", "Date must be in YYYY-MM-DD format.");
      return;
    }

    setSavingEvent(true);
    try {
      const colPath = `academic_events/${uid}/items`;
      await firestoreService.add(colPath, {
        title: title.trim(),
        type,
        date: date.trim(),
        description: description.trim() || undefined
      });
      setTitle("");
      setDate("");
      setDescription("");
      setShowAddForm(false);
      Alert.alert("Success", "Academic milestone added successfully!");
    } catch (err: any) {
      Alert.alert("Error", `Failed to save event: ${err.message}`);
    } finally {
      setSavingEvent(false);
    }
  };

  const handleDeleteEvent = (eventId: string | undefined) => {
    if (!eventId) return;
    Alert.alert(
      "Delete Event",
      "Are you sure you want to remove this academic event?",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Delete",
          style: "destructive",
          onPress: async () => {
            try {
              await firestoreService.delete(`academic_events/${uid}/items/${eventId}`);
            } catch (err: any) {
              Alert.alert("Error", `Failed to delete: ${err.message}`);
            }
          }
        }
      ]
    );
  };

  const handleConfirmEvent = async (eventId: string | undefined) => {
    if (!eventId) return;
    try {
      await firestoreService.update(`academic_events/${uid}/items/${eventId}`, {
        confidence: "user_confirmed"
      });
      Alert.alert("Success", "Milestone confirmed! Study blocks will now be scheduled.");
    } catch (err: any) {
      Alert.alert("Error", `Failed to confirm milestone: ${err.message}`);
    }
  };


  const handlePickTimetableImage = async () => {
    try {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== "granted") {
        Alert.alert("Permission Required", "Need photo library access to upload a screenshot.");
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        quality: 0.6,
      });

      if (result.canceled) return;
      const asset = result.assets[0];

      // File size guard
      const fileInfo = await FileSystem.getInfoAsync(asset.uri);
      if (fileInfo.exists && fileInfo.size && fileInfo.size > 5 * 1024 * 1024) {
        Alert.alert("Error", "Screenshot size exceeds the 5MB limit.");
        return;
      }

      setUploading("timetable");

      const formData = new FormData();
      formData.append("user_id", uid);
      formData.append("file", {
        uri: asset.uri,
        name: asset.fileName || "timetable.jpg",
        type: asset.mimeType || "image/jpeg"
      } as any);

      const response = await fetch(`${API_URL}/api/upload/timetable`, {
        method: "POST",
        headers: await authHeaders({
          "Accept": "application/json",
          "Content-Type": "multipart/form-data",
        }),
        body: formData
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `HTTP ${response.status}`);
      }

      const resJson = await response.json();
      const timetable = resJson.timetable || {};
      const weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
      const normalizedTimetable: Record<string, Array<{time: string, subject: string, room: string, professor: string}>> = {};
      weekdays.forEach(day => {
        normalizedTimetable[day] = timetable[day] || [];
      });
      setTimetableData(normalizedTimetable);
      setRawTimetableJson(JSON.stringify(normalizedTimetable, null, 2));
      setIsEditingTimetable(false);
      setShowTimetableModal(true);
    } catch (err: any) {
      console.error(err);
      Alert.alert("Upload Failed", err.message || "Failed to process timetable screenshot.");
    } finally {
      setUploading(null);
    }
  };

  const updateClassField = (day: string, idx: number, field: string, val: string) => {
    setTimetableData(prev => {
      const updated = { ...prev };
      const dayClasses = [...(updated[day] || [])];
      dayClasses[idx] = { ...dayClasses[idx], [field]: val };
      updated[day] = dayClasses;
      return updated;
    });
  };

  const addClass = (day: string) => {
    setTimetableData(prev => {
      const updated = { ...prev };
      const dayClasses = [...(updated[day] || [])];
      dayClasses.push({ time: "", subject: "", room: "", professor: "" });
      updated[day] = dayClasses;
      return updated;
    });
  };

  const removeClass = (day: string, idx: number) => {
    setTimetableData(prev => {
      const updated = { ...prev };
      const dayClasses = [...(updated[day] || [])];
      dayClasses.splice(idx, 1);
      updated[day] = dayClasses;
      return updated;
    });
  };

  const handlePickSyllabusPdf = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: "application/pdf",
        copyToCacheDirectory: true,
      });

      if (result.canceled) return;
      const asset = result.assets[0];

      if (asset.size && asset.size > 5 * 1024 * 1024) {
        Alert.alert("Error", "File too large. Please upload a PDF under 5MB.");
        return;
      }

      setSyllabusFile({
        uri: asset.uri,
        name: asset.name || "syllabus.pdf",
        type: asset.mimeType || "application/pdf"
      });
      setSyllabusSubject("");
      setShowSubjectPrompt(true);
    } catch (err: any) {
      console.error(err);
      Alert.alert("File Selection Failed", "Failed to select document.");
    }
  };

  const handleUploadSyllabus = async () => {
    if (!syllabusSubject.trim()) {
      Alert.alert("Error", "Please enter a subject name.");
      return;
    }
    if (!syllabusFile) {
      Alert.alert("Error", "No PDF file selected.");
      return;
    }
    setShowSubjectPrompt(false);
    setUploading("syllabus");

    try {
      const formData = new FormData();
      formData.append("user_id", uid);
      formData.append("subject", syllabusSubject.trim());
      formData.append("file", {
        uri: syllabusFile.uri,
        name: syllabusFile.name || "syllabus.pdf",
        type: syllabusFile.type || "application/pdf"
      } as any);

      const response = await fetch(`${API_URL}/api/upload/syllabus`, {
        method: "POST",
        headers: await authHeaders({
          "Accept": "application/json",
          "Content-Type": "multipart/form-data",
        }),
        body: formData
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `HTTP ${response.status}`);
      }

      const resJson = await response.json();
      setRawSyllabusJson(JSON.stringify(resJson.syllabus, null, 2));
      setShowSyllabusModal(true);
    } catch (err: any) {
      console.error(err);
      Alert.alert("Upload Failed", err.message || "Failed to parse syllabus document.");
    } finally {
      setUploading(null);
      setSyllabusFile(null);
    }
  };

  const handleConfirmTimetable = async () => {
    try {
      const cleanedTimetable: Record<string, Array<{time: string, subject: string, room: string, professor: string}>> = {};
      Object.entries(timetableData).forEach(([day, classes]) => {
        cleanedTimetable[day] = classes.filter(cls => cls.subject && cls.subject.trim() !== "");
      });

      const response = await fetch(`${API_URL}/api/upload/confirm/timetable`, {
        method: "POST",
        headers: await authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ user_id: uid, timetable: cleanedTimetable })
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      Alert.alert("Success", "Timetable parsed and saved to database!");
      setShowTimetableModal(false);
      setIsEditingTimetable(false);
    } catch (err: any) {
      Alert.alert("Save Failed", err.message || "Could not confirm timetable details.");
    }
  };

  const handleConfirmSyllabus = async () => {
    try {
      let parsed;
      try {
        parsed = JSON.parse(rawSyllabusJson);
      } catch (jsonErr) {
        Alert.alert("Invalid JSON", "Please ensure your edits result in valid JSON syntax.");
        return;
      }

      const response = await fetch(`${API_URL}/api/upload/confirm/syllabus`, {
        method: "POST",
        headers: await authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({
          user_id: uid,
          subject: parsed.subject,
          syllabus: parsed
        })
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      Alert.alert("Success", `Syllabus for ${parsed.subject} saved successfully!`);
      setShowSyllabusModal(false);
    } catch (err: any) {
      Alert.alert("Save Failed", err.message || "Could not confirm syllabus details.");
    }
  };

  const getDaysRemainingText = (dateStr: string) => {
    const today = new Date();
    today.setHours(0,0,0,0);
    const target = new Date(dateStr);
    target.setHours(0,0,0,0);
    
    const diffTime = target.getTime() - today.getTime();
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays < 0) return "Passed";
    if (diffDays === 0) return "Today! ⚡";
    if (diffDays === 1) return "Tomorrow! 🚨";
    return `${diffDays} days left`;
  };

  const getDaysRemainingColor = (dateStr: string) => {
    const today = new Date();
    today.setHours(0,0,0,0);
    const target = new Date(dateStr);
    target.setHours(0,0,0,0);
    
    const diffTime = target.getTime() - today.getTime();
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays >= 0 && diffDays <= 7) return COLORS.error;
    return COLORS.muted;
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
      keyboardVerticalOffset={Platform.OS === "ios" ? 100 : 0}
    >
      <ScrollView contentContainerStyle={styles.scrollContainer} keyboardShouldPersistTaps="handled">
        {/* DOCUMENT UPLOAD SECTION */}
        <View style={styles.uploadHeaderSection}>
          <Text style={styles.sectionLabel}>📁 Document Parsing & Sync</Text>
          <Text style={styles.sectionSubtitle}>
            Upload your timetable screenshot or syllabus PDF. Gemini will extract, structure, and synchronize them automatically.
          </Text>
        </View>

        <View style={styles.uploadRow}>
          <TouchableOpacity
            style={[styles.uploadCard, uploading === "timetable" && styles.uploadCardActive]}
            onPress={handlePickTimetableImage}
            disabled={uploading !== null}
          >
            {uploading === "timetable" ? (
              <View style={{ alignItems: "center" }}>
                <ActivityIndicator size="small" color={COLORS.accent} />
                <Text style={[styles.uploadCardProgressText, { marginTop: 8 }]}>Reading timetable...</Text>
              </View>
            ) : (
              <>
                <View style={styles.uploadIconCircle}>
                  <Text style={styles.uploadCardIcon}>📅</Text>
                </View>
                <Text style={styles.uploadCardTitle}>Timetable Screenshot</Text>
                <Text style={styles.uploadCardSub}>Upload image (max 5MB)</Text>
              </>
            )}
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.uploadCard, uploading === "syllabus" && styles.uploadCardActive]}
            onPress={handlePickSyllabusPdf}
            disabled={uploading !== null}
          >
            {uploading === "syllabus" ? (
              <View style={{ alignItems: "center" }}>
                <ActivityIndicator size="small" color={COLORS.accent} />
                <Text style={[styles.uploadCardProgressText, { marginTop: 8 }]}>Reading syllabus...</Text>
              </View>
            ) : (
              <>
                <View style={styles.uploadIconCircle}>
                  <Text style={styles.uploadCardIcon}>📖</Text>
                </View>
                <Text style={styles.uploadCardTitle}>Syllabus Document</Text>
                <Text style={styles.uploadCardSub}>Upload PDF (max 5MB)</Text>
              </>
            )}
          </TouchableOpacity>
        </View>

        {/* SUBJECT PROMPT MODAL */}
        <Modal
          visible={showSubjectPrompt}
          transparent={true}
          animationType="fade"
        >
          <View style={styles.modalOverlay}>
            <View style={styles.promptContainer}>
              <Text style={styles.promptHeader}>Target Subject</Text>
              <Text style={styles.promptSub}>Type the course/subject name exactly (e.g. Operating Systems, DBMS):</Text>
              <TextInput
                style={styles.promptInput}
                value={syllabusSubject}
                onChangeText={setSyllabusSubject}
                placeholder="e.g. Operating Systems"
                placeholderTextColor={COLORS.textDim}
              />
              <View style={styles.promptButtons}>
                <TouchableOpacity
                  style={styles.promptBtnSecondary}
                  onPress={() => {
                    setShowSubjectPrompt(false);
                    setSyllabusFile(null);
                  }}
                >
                  <Text style={styles.btnSecondaryText}>Cancel</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.promptBtnPrimary}
                  onPress={handleUploadSyllabus}
                >
                  <Text style={styles.btnPrimaryText}>Extract</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </Modal>

        {/* TIMETABLE CONFIRMATION MODAL */}
        <Modal
          visible={showTimetableModal}
          transparent={true}
          animationType="slide"
        >
          <View style={styles.modalOverlay}>
            <KeyboardAvoidingView
              behavior={Platform.OS === "ios" ? "padding" : "height"}
              style={styles.confirmContainer}
            >
              <View style={{ flex: 1, justifyContent: "space-between" }}>
                <View style={{ marginBottom: 12 }}>
                  <Text style={styles.confirmHeader}>Confirm Extracted Timetable</Text>
                  <Text style={styles.confirmSub}>
                    {isEditingTimetable 
                      ? "Edit the classes below. Leave subject empty to delete/ignore." 
                      : "Verify the extracted classes for each weekday:"}
                  </Text>
                </View>

                <ScrollView style={{ flex: 1, marginBottom: 16 }} keyboardShouldPersistTaps="handled">
                  {Object.entries(timetableData).map(([day, classes]) => {
                    // Only show day in read-only mode if it has classes, but show all in edit mode
                    if (!isEditingTimetable && (!classes || classes.length === 0)) return null;

                    return (
                      <View key={day} style={styles.daySection}>
                        <Text style={styles.dayHeader}>{day}</Text>
                        
                        {classes && classes.map((cls, idx) => (
                          <View key={idx} style={isEditingTimetable ? styles.classEditCard : styles.classCard}>
                            {isEditingTimetable ? (
                              <>
                                <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                                  <Text style={styles.classLabelText}>Class #{idx + 1}</Text>
                                  <TouchableOpacity onPress={() => removeClass(day, idx)}>
                                    <Text style={{ color: COLORS.red, fontSize: 11, fontWeight: "700" }}>Delete 🗑️</Text>
                                  </TouchableOpacity>
                                </View>
                                <TextInput
                                  style={styles.classInput}
                                  value={cls.subject}
                                  onChangeText={(val) => updateClassField(day, idx, "subject", val)}
                                  placeholder="Subject (e.g. DAA)"
                                  placeholderTextColor={COLORS.textDim}
                                />
                                <View style={{ flexDirection: "row", gap: 8 }}>
                                  <TextInput
                                    style={[styles.classInput, { flex: 1 }]}
                                    value={cls.time}
                                    onChangeText={(val) => updateClassField(day, idx, "time", val)}
                                    placeholder="Time (e.g. 09:00)"
                                    placeholderTextColor={COLORS.textDim}
                                  />
                                  <TextInput
                                    style={[styles.classInput, { flex: 1 }]}
                                    value={cls.room}
                                    onChangeText={(val) => updateClassField(day, idx, "room", val)}
                                    placeholder="Room (e.g. LH-201)"
                                    placeholderTextColor={COLORS.textDim}
                                  />
                                  <TextInput
                                    style={[styles.classInput, { flex: 1 }]}
                                    value={cls.professor}
                                    onChangeText={(val) => updateClassField(day, idx, "professor", val)}
                                    placeholder="Prof (e.g. Dr. Roy)"
                                    placeholderTextColor={COLORS.textDim}
                                  />
                                </View>
                              </>
                            ) : (
                              <View>
                                <Text style={[styles.classText, { fontWeight: "700", color: COLORS.accent }]}>
                                  {cls.subject || "Untitled Class"}
                                </Text>
                                <Text style={[styles.classText, { color: COLORS.textMuted, marginTop: 4 }]}>
                                  ⏰ {cls.time || "No time"} | 📍 {cls.room || "No room"} | 👤 {cls.professor || "No prof"}
                                </Text>
                              </View>
                            )}
                          </View>
                        ))}

                        {isEditingTimetable && (
                          <TouchableOpacity 
                            onPress={() => addClass(day)} 
                            style={styles.addClassBtn}
                          >
                            <Text style={styles.addClassBtnText}>
                              + Add Class to {day}
                            </Text>
                          </TouchableOpacity>
                        )}
                      </View>
                    );
                  })}
                </ScrollView>

                <View style={styles.confirmButtons}>
                  {!isEditingTimetable ? (
                    <>
                      <TouchableOpacity
                        style={[styles.confirmBtnSecondary, { marginRight: "auto" }]}
                        onPress={() => setShowTimetableModal(false)}
                      >
                        <Text style={styles.btnSecondaryText}>Close</Text>
                      </TouchableOpacity>
                      <TouchableOpacity
                        style={[styles.confirmBtnSecondary, { backgroundColor: "rgba(255, 138, 155, 0.1)", borderRadius: 10, paddingHorizontal: 14 }]}
                        onPress={() => setIsEditingTimetable(true)}
                      >
                        <Text style={[styles.btnSecondaryText, { color: COLORS.red }]}>Edit Details ✏️</Text>
                      </TouchableOpacity>
                      <TouchableOpacity
                        style={styles.confirmBtnPrimary}
                        onPress={handleConfirmTimetable}
                      >
                        <Text style={styles.btnPrimaryText}>Confirm ✓</Text>
                      </TouchableOpacity>
                    </>
                  ) : (
                    <>
                      <TouchableOpacity
                        style={styles.confirmBtnSecondary}
                        onPress={() => setIsEditingTimetable(false)}
                      >
                        <Text style={styles.btnSecondaryText}>Cancel</Text>
                      </TouchableOpacity>
                      <TouchableOpacity
                        style={styles.confirmBtnPrimary}
                        onPress={handleConfirmTimetable}
                      >
                        <Text style={styles.btnPrimaryText}>Confirm Changes ✓</Text>
                      </TouchableOpacity>
                    </>
                  )}
                </View>
              </View>
            </KeyboardAvoidingView>
          </View>
        </Modal>

        {/* SYLLABUS CONFIRMATION MODAL */}
        <Modal
          visible={showSyllabusModal}
          transparent={true}
          animationType="slide"
        >
          <View style={styles.modalOverlay}>
            <KeyboardAvoidingView
              behavior={Platform.OS === "ios" ? "padding" : "height"}
              style={styles.confirmContainer}
            >
              <Text style={styles.confirmHeader}>Confirm Extracted Syllabus</Text>
              <Text style={styles.confirmSub}>Review syllabus structure and units content. Make manual tweaks to JSON text if needed:</Text>
              
              <TextInput
                style={styles.jsonInput}
                value={rawSyllabusJson}
                onChangeText={setRawSyllabusJson}
                multiline={true}
                autoCapitalize="none"
                autoCorrect={false}
              />

              <View style={styles.confirmButtons}>
                <TouchableOpacity
                  style={styles.confirmBtnSecondary}
                  onPress={() => setShowSyllabusModal(false)}
                >
                  <Text style={styles.btnSecondaryText}>Cancel</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.confirmBtnPrimary}
                  onPress={handleConfirmSyllabus}
                >
                  <Text style={styles.btnPrimaryText}>Save Syllabus</Text>
                </TouchableOpacity>
              </View>
            </KeyboardAvoidingView>
          </View>
        </Modal>

        {/* MANUAL MILESTONE ADDITION */}
        <View style={styles.sectionHeaderRow}>
          <View>
            <Text style={styles.sectionLabel}>🎓 Academic Milestones</Text>
            <Text style={styles.sectionSubtitleText}>Assignments, exams, and grading milestones</Text>
          </View>
          <TouchableOpacity
            style={styles.addButton}
            onPress={() => setShowAddForm(!showAddForm)}
          >
            <Text style={styles.addButtonText}>
              {showAddForm ? "Close" : "+ Add Milestone"}
            </Text>
          </TouchableOpacity>
        </View>

        {showAddForm && (
          <View style={styles.addForm}>
            <Text style={styles.formTitle}>Add Milestone</Text>
            
            <View style={styles.formInputGroup}>
              <Text style={styles.inputLabel}>Title</Text>
              <TextInput
                style={styles.textInput}
                value={title}
                onChangeText={setTitle}
                placeholder="e.g. CIE-1: Database Systems"
                placeholderTextColor={COLORS.textDim}
              />
            </View>

            <View style={styles.formInputGroup}>
              <Text style={styles.inputLabel}>Due Date (YYYY-MM-DD)</Text>
              <TextInput
                style={styles.textInput}
                value={date}
                onChangeText={setDate}
                placeholder="e.g. 2026-06-25"
                placeholderTextColor={COLORS.textDim}
              />
            </View>

            <View style={styles.formInputGroup}>
              <Text style={styles.inputLabel}>Milestone Type</Text>
              <View style={styles.typeSelector}>
                {["cie", "see", "assignment", "lab_record", "viva"].map(t => {
                  const isSelected = type === t;
                  let activeBg = COLORS.accentBg;
                  let activeBorder = COLORS.accent;
                  let activeText = COLORS.accent;
                  
                  if (t === "see") {
                    activeBg = "rgba(255, 138, 155, 0.12)";
                    activeBorder = COLORS.red;
                    activeText = COLORS.red;
                  } else if (t === "assignment") {
                    activeBg = COLORS.lavenderBg;
                    activeBorder = COLORS.lavender;
                    activeText = COLORS.lavender;
                  } else if (t === "lab_record") {
                    activeBg = "rgba(110, 231, 168, 0.12)";
                    activeBorder = COLORS.green;
                    activeText = COLORS.green;
                  } else if (t === "viva") {
                    activeBg = "rgba(127, 182, 255, 0.12)";
                    activeBorder = COLORS.blue;
                    activeText = COLORS.blue;
                  }

                  return (
                    <TouchableOpacity
                      key={t}
                      style={[
                        styles.typeButton, 
                        isSelected && { backgroundColor: activeBg, borderColor: activeBorder }
                      ]}
                      onPress={() => setType(t as any)}
                    >
                      <Text style={[
                        styles.typeButtonText, 
                        isSelected && { color: activeText, fontWeight: "700" }
                      ]}>
                        {t.toUpperCase().replace("_", " ")}
                      </Text>
                    </TouchableOpacity>
                  );
                })}
              </View>
            </View>

            <View style={styles.formInputGroup}>
              <Text style={styles.inputLabel}>Description</Text>
              <TextInput
                style={[styles.textInput, styles.textArea]}
                value={description}
                onChangeText={setDescription}
                placeholder="Details (modules covered, grading weights, etc.)"
                placeholderTextColor={COLORS.textDim}
                multiline={true}
              />
            </View>

            <TouchableOpacity
              style={styles.submitButton}
              onPress={handleAddEvent}
              disabled={savingEvent}
            >
              {savingEvent ? (
                <ActivityIndicator size="small" color="#0c0e14" />
              ) : (
                <Text style={styles.submitButtonText}>Save Milestone</Text>
              )}
            </TouchableOpacity>
          </View>
        )}

        {/* EVENTS LIST */}
        {loadingEvents ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="small" color={COLORS.accent} />
          </View>
        ) : events.length === 0 ? (
          <View style={styles.emptyCard}>
            <Text style={styles.emptyText}>No academic milestones registered yet. Fill them in above, upload a document, or tell the AI assistant.</Text>
          </View>
        ) : (
          events.map(event => {
            const daysColor = getDaysRemainingColor(event.date);
            const daysText = getDaysRemainingText(event.date);

            let eventColor = COLORS.accent;
            let eventTypeBg = COLORS.accentBg;
            if (event.type === "cie") {
              eventColor = COLORS.accent;
              eventTypeBg = COLORS.accentBg;
            } else if (event.type === "see") {
              eventColor = COLORS.red;
              eventTypeBg = "rgba(255, 138, 155, 0.12)";
            } else if (event.type === "assignment") {
              eventColor = COLORS.lavender;
              eventTypeBg = COLORS.lavenderBg;
            } else if (event.type === "lab_record") {
              eventColor = COLORS.green;
              eventTypeBg = "rgba(110, 231, 168, 0.12)";
            } else if (event.type === "viva") {
              eventColor = COLORS.blue;
              eventTypeBg = "rgba(127, 182, 255, 0.12)";
            }
            
            return (
              <View 
                key={event.id} 
                style={[
                  styles.eventCard, 
                  { borderLeftColor: eventColor },
                  event.confidence === "scraped" && styles.eventCardScraped
                ]}
              >
                <View style={styles.eventHeader}>
                  <View style={styles.eventTitleCol}>
                    <Text style={styles.eventTitle}>{event.title}</Text>
                    <View style={[styles.badgeContainer, { backgroundColor: eventTypeBg }]}>
                      <Text style={[styles.badgeText, { color: eventColor }]}>{event.type.toUpperCase()}</Text>
                    </View>
                    {event.confidence === "scraped" && (
                      <View style={[styles.badgeContainer, { backgroundColor: "rgba(255, 212, 121, 0.12)" }]}>
                        <Text style={[styles.badgeText, { color: COLORS.yellow }]}>⚠️ TENTATIVE</Text>
                      </View>
                    )}
                  </View>
                  <TouchableOpacity
                    style={styles.deleteMilestoneButton}
                    onPress={() => handleDeleteEvent(event.id)}
                  >
                    <Text style={styles.deleteMilestoneButtonText}>✕</Text>
                  </TouchableOpacity>
                </View>

                {event.description ? (
                  <Text style={styles.eventDesc}>{event.description}</Text>
                ) : null}

                <View style={styles.eventFooter}>
                  <View style={{ flexDirection: "row", alignItems: "center", gap: 12 }}>
                    <Text style={styles.eventDate}>📅 {event.date}</Text>
                    {event.confidence === "scraped" && (
                      <TouchableOpacity
                        onPress={() => handleConfirmEvent(event.id)}
                        style={styles.confirmEventBtn}
                      >
                        <Text style={styles.confirmEventBtnText}>Confirm Date</Text>
                      </TouchableOpacity>
                    )}
                  </View>
                  <Text style={[styles.daysLeft, { color: daysColor }]}>{daysText}</Text>
                </View>
              </View>
            );
          })
        )}
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.bg
  },
  scrollContainer: {
    padding: 20,
    paddingBottom: 40
  },
  sectionLabel: {
    color: COLORS.text,
    fontSize: 18,
    fontWeight: "800",
    marginBottom: 6
  },
  sectionSubtitle: {
    color: COLORS.textDim,
    fontSize: 12,
    lineHeight: 18,
    marginBottom: 16
  },
  uploadHeaderSection: {
    marginBottom: 16
  },
  uploadRow: {
    flexDirection: "row",
    gap: 12,
    marginBottom: 24
  },
  uploadCard: {
    flex: 1,
    backgroundColor: COLORS.surface,
    borderWidth: 1.5,
    borderStyle: "dashed",
    borderColor: COLORS.border,
    borderRadius: 16,
    padding: 16,
    alignItems: "center",
    justifyContent: "center",
    minHeight: 140
  },
  uploadCardActive: {
    borderColor: COLORS.accent,
    backgroundColor: COLORS.accentBg
  },
  uploadIconCircle: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: COLORS.panel,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 10
  },
  uploadCardIcon: {
    fontSize: 18
  },
  uploadCardTitle: {
    color: COLORS.text,
    fontSize: 12,
    fontWeight: "700",
    textAlign: "center",
    marginBottom: 4
  },
  uploadCardSub: {
    color: COLORS.textDim,
    fontSize: 10,
    textAlign: "center"
  },
  uploadCardProgressText: {
    color: COLORS.accent,
    fontSize: 11,
    fontWeight: "700"
  },
  sectionHeaderRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 16,
    marginTop: 8
  },
  sectionSubtitleText: {
    color: COLORS.textDim,
    fontSize: 12,
    marginTop: 2
  },
  addButton: {
    backgroundColor: COLORS.accentBg,
    borderWidth: 1,
    borderColor: COLORS.accentLine,
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 6
  },
  addButtonText: {
    color: COLORS.accent,
    fontSize: 11,
    fontWeight: "700"
  },
  addForm: {
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 16,
    padding: 16,
    marginBottom: 20
  },
  formTitle: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: "700",
    marginBottom: 12
  },
  formInputGroup: {
    marginBottom: 12
  },
  inputLabel: {
    color: COLORS.textMuted,
    fontSize: 11,
    fontWeight: "600",
    marginBottom: 6
  },
  textInput: {
    backgroundColor: COLORS.bg,
    color: COLORS.text,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 13,
  },
  typeSelector: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 6,
    marginTop: 4
  },
  typeButton: {
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 6,
    backgroundColor: COLORS.panel
  },
  typeButtonText: {
    color: COLORS.textDim,
    fontSize: 9,
    fontFamily: "monospace",
    fontWeight: "700"
  },
  textArea: {
    minHeight: 80,
    textAlignVertical: "top"
  },
  submitButton: {
    backgroundColor: COLORS.accent,
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: "center",
    marginTop: 8
  },
  submitButtonText: {
    color: "#0c0e14",
    fontSize: 13,
    fontWeight: "700"
  },
  loadingContainer: {
    paddingVertical: 20,
    alignItems: "center"
  },
  emptyCard: {
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 16,
    padding: 24,
    alignItems: "center"
  },
  emptyText: {
    color: COLORS.textMuted,
    fontSize: 12,
    lineHeight: 18,
    textAlign: "center"
  },
  eventCard: {
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderLeftWidth: 4,
    borderRadius: 16,
    padding: 16,
    marginBottom: 12
  },
  eventCardScraped: {
    borderStyle: "dashed"
  },
  eventHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: 8
  },
  eventTitleCol: {
    flex: 1,
    flexDirection: "row",
    flexWrap: "wrap",
    alignItems: "center",
    gap: 8
  },
  eventTitle: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: "700"
  },
  badgeContainer: {
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4
  },
  badgeText: {
    fontSize: 8,
    fontFamily: "monospace",
    fontWeight: "700"
  },
  deleteMilestoneButton: {
    padding: 6
  },
  deleteMilestoneButtonText: {
    color: COLORS.textDim,
    fontSize: 13,
    fontWeight: "600"
  },
  eventDesc: {
    color: COLORS.textMuted,
    fontSize: 12,
    lineHeight: 18,
    marginBottom: 12
  },
  eventFooter: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
    paddingTop: 10
  },
  eventDate: {
    color: COLORS.text,
    fontSize: 11,
    fontFamily: "monospace"
  },
  confirmEventBtn: {
    backgroundColor: COLORS.accent,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8
  },
  confirmEventBtnText: {
    color: "#0c0e14",
    fontSize: 10,
    fontWeight: "700"
  },
  daysLeft: {
    fontSize: 11,
    fontFamily: "monospace",
    fontWeight: "700"
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0, 0, 0, 0.75)",
    justifyContent: "center",
    alignItems: "center",
    padding: 20
  },
  promptContainer: {
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 20,
    padding: 20,
    width: "100%",
    maxWidth: 320
  },
  promptHeader: {
    color: COLORS.text,
    fontSize: 16,
    fontWeight: "700",
    marginBottom: 8
  },
  promptSub: {
    color: COLORS.textMuted,
    fontSize: 12,
    lineHeight: 18,
    marginBottom: 16
  },
  promptInput: {
    backgroundColor: COLORS.bg,
    color: COLORS.text,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 13,
    marginBottom: 20
  },
  promptButtons: {
    flexDirection: "row",
    justifyContent: "flex-end",
    gap: 12
  },
  promptBtnSecondary: {
    paddingVertical: 8,
    paddingHorizontal: 16
  },
  promptBtnPrimary: {
    backgroundColor: COLORS.accent,
    borderRadius: 8,
    paddingVertical: 8,
    paddingHorizontal: 16
  },
  btnSecondaryText: {
    color: COLORS.text,
    fontSize: 13,
    fontWeight: "600"
  },
  btnPrimaryText: {
    color: "#0c0e14",
    fontSize: 13,
    fontWeight: "700"
  },
  confirmContainer: {
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 20,
    padding: 20,
    width: "100%",
    height: "90%",
    justifyContent: "space-between"
  },
  confirmHeader: {
    color: COLORS.text,
    fontSize: 16,
    fontWeight: "700"
  },
  confirmSub: {
    color: COLORS.textMuted,
    fontSize: 12,
    lineHeight: 16,
    marginTop: 4,
    marginBottom: 12
  },
  jsonInput: {
    flex: 1,
    backgroundColor: COLORS.bg,
    color: COLORS.accent,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 12,
    padding: 12,
    fontFamily: "monospace",
    fontSize: 11,
    textAlignVertical: "top",
    marginBottom: 16
  },
  confirmButtons: {
    flexDirection: "row",
    justifyContent: "flex-end",
    gap: 12
  },
  confirmBtnSecondary: {
    paddingVertical: 10,
    paddingHorizontal: 16
  },
  confirmBtnPrimary: {
    backgroundColor: COLORS.accent,
    borderRadius: 8,
    paddingVertical: 10,
    paddingHorizontal: 20
  },
  daySection: {
    marginBottom: 16
  },
  dayHeader: {
    color: COLORS.accent,
    fontSize: 14,
    fontWeight: "700",
    marginBottom: 8,
    fontFamily: "monospace"
  },
  classCard: {
    backgroundColor: COLORS.bg,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 8,
    padding: 10,
    marginBottom: 8
  },
  classText: {
    color: COLORS.text,
    fontSize: 12,
    fontFamily: "monospace"
  },
  classEditCard: {
    backgroundColor: COLORS.bg,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 8,
    padding: 10,
    marginBottom: 8,
    gap: 6
  },
  classInput: {
    backgroundColor: COLORS.surface,
    color: COLORS.text,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 6,
    fontSize: 11,
    fontFamily: "monospace"
  },
  addClassBtn: {
    borderWidth: 1,
    borderColor: COLORS.accentLine,
    borderStyle: "dashed",
    backgroundColor: COLORS.accentBg,
    borderRadius: 10,
    padding: 10,
    alignItems: "center",
    marginTop: 8
  },
  addClassBtnText: {
    color: COLORS.accent,
    fontSize: 11,
    fontWeight: "700"
  },
  classLabelText: {
    color: COLORS.textMuted,
    fontSize: 11,
    fontWeight: "700"
  }
});

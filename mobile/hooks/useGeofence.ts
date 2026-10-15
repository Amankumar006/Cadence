import * as Location from "expo-location";
import * as TaskManager from "expo-task-manager";
import { authService } from "../services/firebase";
import { API_URL, authHeaders } from "../services/apiHeaders";

export const GEOFENCE_TASK_NAME = "GEOFENCE_BACKGROUND_TASK";

async function triggerLocationArrival(regionName: string) {
  try {
    const user = authService.currentUser;
    if (!user) {
      console.warn("No user logged in, cannot trigger location arrival");
      return;
    }
    const apiUrl = API_URL;
    
    // Map identifier to the exact phrase the backend expects
    let messageText = `I just arrived at the ${regionName}`;
    if (regionName === "home") {
      messageText = "I just arrived at home";
    } else if (regionName === "gym") {
      messageText = "I just arrived at the gym";
    } else if (regionName === "college") {
      messageText = "I just arrived at college";
    }
    
    console.info(`Geofence background trigger posting to ${apiUrl}/api/chat: "${messageText}"`);

    const response = await fetch(`${apiUrl}/api/chat`, {
      method: "POST",
      headers: await authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        user_id: user.uid,
        thread_id: "default-chat-thread",
        message: messageText,
      }),
    });

    if (!response.ok) {
      console.error(`Geofence POST failed: ${response.status} ${response.statusText}`);
    } else {
      console.info("Geofence POST succeeded");
    }
  } catch (err) {
    console.error("Error sending geofence trigger to backend:", err);
  }
}

// Register the background task
TaskManager.defineTask(GEOFENCE_TASK_NAME, async ({ data, error }: any) => {
  if (error) {
    console.error("Geofence background task error:", error);
    return;
  }
  
  const eventType = data?.eventType;
  const region = data?.region;
  
  console.info(`Geofence background task triggered: eventType=${eventType}, region=${JSON.stringify(region)}`);
  
  if (eventType === Location.GeofencingEventType.Enter && region?.identifier) {
    await triggerLocationArrival(region.identifier);
  }
});

export async function setupGeofences(locations: {
  gym?: { latitude: number; longitude: number };
  college?: { latitude: number; longitude: number };
  home?: { latitude: number; longitude: number };
}) {
  try {
    const isRegistered = await TaskManager.isTaskRegisteredAsync(GEOFENCE_TASK_NAME);
    
    // Create regions array based on locations
    const regions: Location.LocationRegion[] = [];
    
    if (locations?.gym?.latitude && locations?.gym?.longitude) {
      regions.push({
        identifier: "gym",
        latitude: locations.gym.latitude,
        longitude: locations.gym.longitude,
        radius: 100, // 100 meters
        notifyOnEnter: true,
        notifyOnExit: false,
      });
    }
    if (locations?.college?.latitude && locations?.college?.longitude) {
      regions.push({
        identifier: "college",
        latitude: locations.college.latitude,
        longitude: locations.college.longitude,
        radius: 150, // 150 meters
        notifyOnEnter: true,
        notifyOnExit: false,
      });
    }
    if (locations?.home?.latitude && locations?.home?.longitude) {
      regions.push({
        identifier: "home",
        latitude: locations.home.latitude,
        longitude: locations.home.longitude,
        radius: 50, // 50 meters
        notifyOnEnter: true,
        notifyOnExit: false,
      });
    }

    if (regions.length === 0) {
      console.info("No geofencing regions configured.");
      if (isRegistered) {
        await Location.stopGeofencingAsync(GEOFENCE_TASK_NAME);
        console.info("Geofencing stopped since regions are empty.");
      }
      return;
    }

    // Request foreground location permissions
    const { status: foregroundStatus } = await Location.requestForegroundPermissionsAsync();
    if (foregroundStatus !== "granted") {
      console.warn("Foreground location permission not granted, cannot start geofencing.");
      return;
    }

    // Request background location permissions
    const { status: backgroundStatus } = await Location.requestBackgroundPermissionsAsync();
    if (backgroundStatus !== "granted") {
      console.warn("Background location permission not granted, cannot start geofencing.");
      return;
    }

    console.info(`Starting geofencing with ${regions.length} regions:`, JSON.stringify(regions));
    await Location.startGeofencingAsync(GEOFENCE_TASK_NAME, regions);
    console.info("Geofencing started successfully.");
  } catch (err) {
    console.error("Error setting up geofences:", err);
  }
}

// Manual trigger for testing and validation
export async function mockTriggerGeofence(regionName: "gym" | "college" | "home") {
  console.info(`Mocking geofence enter event for: ${regionName}`);
  await triggerLocationArrival(regionName);
}

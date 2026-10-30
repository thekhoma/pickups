// Names a sentence can use. Each word points at a stand-in we can actually draw.
// The picture stays a small gallery. The list is what makes a niche word resolve.

export interface Tag {
  place?: string;
  who?: "person" | "dog" | "cat";
  child?: boolean;
  ride?: "car" | "bike" | "skate" | "moto" | "boat";
  prop?: "bench" | "cup" | "phone" | "tree" | "house" | "fountain" | "table" | "post" | "rock";
  motion?: "sit" | "walk" | "jog" | "run" | "ride" | "jump" | "lay" | "kneel" | "crouch" | "wave";
  city?: "sidewalk" | "alley" | "plaza";
  label: string;
}

const TOKEN = new Map<string, Tag>();
const PHRASES: { phrase: string; tag: Tag }[] = [];

function add(words: string, tag: Tag) {
  for (const word of words.split(/\s+/)) {
    if (word && !TOKEN.has(word)) TOKEN.set(word, tag);
  }
}

function phrase(text: string, tag: Tag) {
  PHRASES.push({ phrase: text, tag });
}

add(
  "interior indoors inside kitchen bedroom bathroom office cafe restaurant diner lobby hallway classroom apartment loft bar cafeteria library museum theater theatre cinema gym studio salon bakery restroom attic basement pantry closet",
  { place: "interior", label: "Interior" },
);
add("city downtown street streets avenue block skyline urban rooftop intersection parking", { place: "city", label: "City" });
add("sidewalk pavement crosswalk curb", { place: "city", city: "sidewalk", label: "Sidewalk" });
add("alley alleyway", { place: "city", city: "alley", label: "Alley" });
add("plaza square market", { place: "city", city: "plaza", label: "Plaza" });
add("suburb suburbs suburban neighborhood neighbourhood driveway lawn culdesac", { place: "suburbs", label: "Suburbs" });
add("outskirts highway freeway roadside industrial warehouse farm field rural countryside ranch meadow pasture village", { place: "outskirts", label: "Outskirts" });
add("desert dunes canyon mesa sand wasteland", { place: "desert", label: "Desert" });
add("forest woods woodland jungle trail mountain mountains hill hills pine grove", { place: "forest", label: "Forest" });
add("ocean beach shore sea coast pier harbor harbour bay marina coastline", { place: "ocean", label: "Ocean" });
add("lake lakeside pond dock", { place: "lake", label: "Lake" });
add("river riverside creek stream waterfall canal brook", { place: "river", label: "River" });
add("park garden playground campus yard backyard courtyard", { place: "park", label: "Park" });
add("house home cabin cottage mansion villa barn", { place: "house", label: "House" });

phrase("living room", { place: "interior", label: "Interior" });
phrase("dining room", { place: "interior", label: "Interior" });
phrase("edge of town", { place: "outskirts", label: "Outskirts" });
phrase("parking lot", { place: "city", label: "City" });
phrase("coffee shop", { place: "interior", label: "Interior" });
phrase("in the house", { place: "interior", label: "Interior" });
phrase("inside a house", { place: "interior", label: "Interior" });
phrase("inside the house", { place: "interior", label: "Interior" });

add(
  "person people man woman guy girl someone somebody figure human pedestrian family boy stranger rider driver passenger friend couple crowd",
  { who: "person", label: "Person" },
);
add(
  "waiter waitress chef cook cop officer worker tourist nurse doctor teacher student artist musician dancer guard soldier knight king queen bride groom",
  { who: "person", label: "Person" },
);
add("kid child children baby toddler", { who: "person", child: true, label: "Child" });
add("dog dogs puppy puppies hound wolf fox", { who: "dog", label: "Dog" });
add("cat cats kitten kittens", { who: "cat", label: "Cat" });

add("car cars vehicle vehicles suv truck van bus taxi sedan jeep pickup", { ride: "car", label: "Car" });
add("bike bicycle bikes bicycles cyclist cycling", { ride: "bike", who: "person", motion: "ride", label: "Bicycle" });
add("skateboard skateboarding skater", { ride: "skate", who: "person", motion: "ride", label: "Skateboard" });
add("motorcycle motorbike scooter", { ride: "moto", who: "person", motion: "ride", label: "Motorcycle" });
add("boat canoe kayak sailboat ship yacht ferry", { ride: "boat", label: "Boat" });

add("bench chair seat stool sofa couch pew", { prop: "bench", label: "Seat" });
add("cup mug coffee tea glass bottle wine beer sip sips sipping drink drinks drinking", { prop: "cup", label: "Cup" });
add("phone cellphone mobile", { prop: "phone", label: "Phone" });
add("tree trees bush plant flower flowers hedge", { prop: "tree", label: "Tree" });
add("building tower shed garage", { prop: "house", label: "Building" });
add("fountain statue monument", { prop: "fountain", label: "Fountain" });
add("table desk counter", { prop: "table", label: "Table" });
add("hydrant mailbox sign lamp bin trash bollard meter post", { prop: "post", label: "Post" });
add("rock rocks boulder stone", { prop: "rock", label: "Rock" });

phrase("coffee cup", { prop: "cup", label: "Cup" });
phrase("park bench", { prop: "bench", place: "park", label: "Seat" });
phrase("fire hydrant", { prop: "post", label: "Post" });
phrase("trash can", { prop: "post", label: "Post" });

add("sit sits sitting seated", { motion: "sit", label: "Sitting" });
add("jump jumps jumping hop hops leap leaps leaping", { motion: "jump", label: "Jumping" });
add("lay lays laying lie lies lying prone asleep", { motion: "lay", label: "Lying" });
add("kneel kneels kneeling", { motion: "kneel", label: "Kneeling" });
add("crouch crouches crouching squat squats squatting", { motion: "crouch", label: "Crouching" });
add("wave waves waving", { motion: "wave", label: "Waving" });
add("walk walks walking", { motion: "walk", label: "Walking" });
add("jog jogging", { motion: "jog", label: "Jogging" });
add("run runs running sprint sprints", { motion: "run", label: "Running" });
add("ride rides riding skate skating drive drives driving", { motion: "ride", label: "Riding" });
add("parked", { label: "Parked" });

export interface Scan {
  place: string | null;
  person: boolean;
  child: boolean;
  dog: boolean;
  cat: boolean;
  car: boolean;
  bike: boolean;
  skate: boolean;
  moto: boolean;
  boat: boolean;
  bench: boolean;
  cup: boolean;
  phone: boolean;
  trees: boolean;
  house: boolean;
  fountain: boolean;
  table: boolean;
  post: boolean;
  rock: boolean;
  sit: boolean;
  walk: boolean;
  gait: "walk" | "jog" | "run" | null;
  action: "jump" | "lay" | "kneel" | "crouch" | "wave" | null;
  city: "sidewalk" | "alley" | "plaza" | null;
  ride: boolean;
  used: Set<string>;
  labels: string[];
}

function blank(): Scan {
  return {
    place: null,
    person: false,
    child: false,
    dog: false,
    cat: false,
    car: false,
    bike: false,
    skate: false,
    moto: false,
    boat: false,
    bench: false,
    cup: false,
    phone: false,
    trees: false,
    house: false,
    fountain: false,
    table: false,
    post: false,
    rock: false,
    sit: false,
    walk: false,
    gait: null,
    action: null,
    city: null,
    ride: false,
    used: new Set(),
    labels: [],
  };
}

function apply(scan: Scan, tag: Tag) {
  if (tag.who === "person") scan.person = true;
  if (tag.child) scan.child = true;
  if (tag.who === "dog") scan.dog = true;
  if (tag.who === "cat") scan.cat = true;
  if (tag.ride === "car") scan.car = true;
  if (tag.ride === "bike") scan.bike = true;
  if (tag.ride === "skate") scan.skate = true;
  if (tag.ride === "moto") scan.moto = true;
  if (tag.ride === "boat") scan.boat = true;
  if (tag.prop === "bench") scan.bench = true;
  if (tag.prop === "cup") scan.cup = true;
  if (tag.prop === "phone") scan.phone = true;
  if (tag.prop === "tree") scan.trees = true;
  if (tag.prop === "house") scan.house = true;
  if (tag.prop === "fountain") scan.fountain = true;
  if (tag.prop === "table") scan.table = true;
  if (tag.prop === "post") scan.post = true;
  if (tag.prop === "rock") scan.rock = true;
  if (tag.motion === "sit") scan.sit = true;
  if (tag.motion === "jump" || tag.motion === "lay" || tag.motion === "kneel" || tag.motion === "crouch" || tag.motion === "wave") scan.action = tag.motion;
  if (tag.motion === "walk" || tag.motion === "jog" || tag.motion === "run") {
    scan.walk = true;
    const rank = { walk: 1, jog: 2, run: 3 };
    if (!scan.gait || rank[tag.motion] >= rank[scan.gait]) scan.gait = tag.motion;
  }
  if (tag.city && !scan.city) scan.city = tag.city;
  if (tag.motion === "ride") scan.ride = true;
  if (tag.label && tag.label !== "Sitting" && tag.label !== "Walking" && tag.label !== "Riding" && tag.label !== "Parked" && !scan.labels.includes(tag.label)) {
    scan.labels.push(tag.label);
  }
}

export function scan(text: string): Scan {
  const out = blank();
  const hits: { index: number; phrase: string; tag: Tag }[] = [];
  for (const entry of PHRASES) {
    let from = 0;
    while (from < text.length) {
      const index = text.indexOf(entry.phrase, from);
      if (index < 0) break;
      const before = index === 0 || text[index - 1] === " ";
      const end = index + entry.phrase.length;
      const after = end === text.length || text[end] === " ";
      if (before && after) hits.push({ index, phrase: entry.phrase, tag: entry.tag });
      from = index + 1;
    }
  }
  let offset = 0;
  for (const token of text.split(" ")) {
    const tag = TOKEN.get(token);
    if (tag) hits.push({ index: offset, phrase: token, tag });
    offset += token.length + 1;
  }
  hits.sort((a, b) => a.index - b.index || b.phrase.length - a.phrase.length);
  const placeHits = hits.filter((hit) => hit.tag.place);
  const locative = placeHits.find((hit) =>
    /\b(?:in|at|by|near|beside|along|through|across|over)\s+(?:the\s+|a\s+|an\s+)?$/.test(text.slice(Math.max(0, hit.index - 18), hit.index)),
  );
  out.place = (locative ?? placeHits[0])?.tag.place ?? null;
  for (const hit of hits) {
    apply(out, hit.tag);
    for (const word of hit.phrase.split(" ")) out.used.add(word);
  }
  return out;
}

export const LEXICON_COUNT = TOKEN.size + PHRASES.length;

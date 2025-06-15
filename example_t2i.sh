# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

# model_path="OmniGen2/OmniGen2"
# python inference.py \
# --model_path $model_path \
# --num_inference_step 50 \
# --height 768 \
# --width 1024 \
# --text_guidance_scale 3.5 \
# --instruction "A curly-haired man in a red shirt is drinking tea." \
# --output_image_path outputs/output_t2i.png \
# --num_images_per_prompt 1

instructions=(
# "Cinematic shot, dramatic lighting. In an ancient throne room where dust motes dance in the light, a pure white Persian cat with heterochromia (one gold eye, one blue eye) wears a heavy, ancient crown that is slightly too large and slipping. It yawns arrogantly, lounging on a deeply creased, slightly faded, dark purple velvet throne. A single god-ray from a high window illuminates it perfectly, highlighting its fluffy fur and the gems on the crown. The scene is filled with a sense of absurd yet solemn power. Ultra-wide angle, 8K, hyper-detailed, narrative."
# "Concept art, dark fantasy style. In a forgotten subterranean grotto, the only light source is a complex chronomancy spell being cast from the wizard's hands—a clock-face-like sigil made of rotating golden gears and blue energy lines. The wizard's face is illuminated from below by the spell, casting sharp shadows and revealing a fanatical, focused expression. Around him, water droplets and shattered rocks float in defiance of gravity, and massive prehistoric fossils embedded in the cave walls seem to stir in the shadows. Trending on Artstation, style of Greg Rutkowski, motion blur, powerful."
# "Ethereal portrait photography, mythological style. A close-up shot of a snow maiden with a single teardrop freezing on her cheek. Her skin has a pearlescent, cold luminescence, as if lit from within. Her frosty white lashes are incredibly long, hung with delicate ice crystals that refract the soft moonlight from the background. Her hair, like a soft snowdrift, merges with the snow-laden branches behind her. The focus is on her ice-blue eyes, which clearly reflect the image of a warm, burning fireplace she can never reach. Shallow depth of field, bokeh, ethereal, beautiful sadness."
# "National Geographic cinematic landscape. An epic, grand-scale scene where a celestial ballet of emerald green and violet aurora ribbons dances and pulsates across a star-filled sky. Below lies a vast, fractured ice sheet, with deep crevasses glowing with an enigmatic blue light from within. At the golden ratio point of the frame stands a solitary, ancient obsidian monolith, its polished surface reflecting the magnificent spectacle above. Low-angle, ultra-wide lens, emphasizing the immense scale of the earth and sky and the mystery of the monolith, 8K, hyper-realistic, awe-inspiring."
# "Cinematic fantasy macro photography. A delicate clockwork ladybug, constructed from polished brass and faceted ruby, is unfurling its fragile silver wings on a velvety, deep crimson rose petal. Directly in its path, a perfectly spherical dewdrop acts like a fisheye lens, refracting and distorting a view of the entire mist-shrouded dawn garden. The first golden rays of dawn penetrate the dewdrop, casting tiny rainbow caustics and glinting off the metallic insect's body. Razor-thin depth of field, beautiful bokeh, magical realism."
# "Cinematic wide shot of a massive, battle-scarred alpha wolf with piercing amber eyes, its thick grey fur is wet and matted. The wolf is cautiously drinking from a rushing, crystal-clear stream in an ancient, mist-shrouded redwood forest. Eerie moonlight filters through the dense canopy, creating dramatic volumetric god rays and long shadows. The atmosphere is tense and primal. Photorealistic, hyper-detailed, 8K resolution, shot on a telephoto lens, professional photography."
# "Hyperrealistic macro photograph of a whimsical rabbit sculpture, meticulously crafted from an assortment of fresh garden vegetables. Its body is formed from crisp lettuce and cabbage leaves, with vibrant carrot slices for ears, bright red radish for eyes, and delicate parsley sprigs for fur. The rabbit is sitting on a rustic, dark wood cutting board, with a few scattered water droplets glistening on its surface. Dramatic, warm studio lighting from the side casts soft shadows, highlighting the intricate textures of the vegetables. Shallow depth of field, sharp focus, cinematic food photography, 8K, bokeh background."
# "Epic fantasy digital painting of a celestial, kaleidoscopic phoenix in mid-flight, its magnificent wings unfurled. Its feathers are made of shimmering, liquid crystal fractals and glowing with vibrant bioluminescent patterns, trailing a nebula of sparkling light particles. The bird soars through an enchanted, alien rainforest at twilight, filled with giant, glowing flora and ancient, mossy trees. Ethereal, volumetric light rays emanate from the bird and the magical plants, creating a mystical and otherworldly atmosphere. Dynamic composition, sense of epic scale and wonder, intricate details, masterpiece, concept art, trending on ArtStation, 8K resolution."

# "Cinematic portrait of a grim and ancient Dwarf King, his face a roadmap of countless battles, with a long, intricately braided silver beard adorned with gold rings. He sits heavily upon a massive, brutalist throne carved from raw obsidian and forged iron, runes glowing faintly with inner fire. The throne hall is a vast, cavernous subterranean chamber, lit by flickering torches that cast long, dancing shadows on colossal stone pillars. A single, powerful beam of light from a high grate illuminates the king, catching the glint of his mythril crown and the wear on his heavy plate armor. Atmosphere of immense age, power, and solitude. Hyperrealistic, fantasy art, 8K, dramatic lighting."
# "Epic cinematic shot of a lone astronaut performing a spacewalk, floating silently in the vast emptiness above the rust-red sphere of Mars. The Martian surface, with its grand canyons and polar ice caps, fills the lower frame. In the far distance, the tiny, brilliant blue marble of Earth hangs like a fragile jewel against the infinite, star-dusted blackness of space. The sun's harsh, unfiltered light creates sharp, high-contrast highlights on the astronaut's detailed suit and a brilliant reflection of the red planet on their visor. A profound sense of awe, isolation, and the immense scale of the cosmos. Ultra-realistic, NASA-style photography, 8K resolution, wide-angle lens."
# "Macro shot of a small, weathered exploration robot, its metallic chassis scratched and dented, with a single, softly glowing blue optic lens. It is cautiously navigating the moss-covered, rain-slicked stones of a forgotten jungle temple ruin. Dappled sunlight filters through the dense, humid canopy above, creating ethereal god rays that illuminate floating dust motes and the intricate textures of the ancient, vine-choked carvings. The atmosphere is one of serene discovery and deep antiquity. Photorealistic, shallow depth of field, cinematic lighting, Unreal Engine 5 render, 8K."
# "Whimsical and magical cinematic close-up of an adorable, fluffy giant panda sitting in a mystical bamboo forest at night. The panda holds a celestial mooncake, crafted from luminous, crystal-clear ice, glowing with a soft, internal light blue light. Delicate frost patterns cover its surface. The moonlight filters through the tall bamboo stalks, casting soft, silvery rim lighting on the panda's fur and making the ice mooncake sparkle. The background is filled with glowing fireflies and a gentle bokeh effect. Hyper-detailed, dreamlike atmosphere, fantasy food art, 8K."
"Dynamic and majestic action shot of a magnificent snowy owl, its massive wingspan fully extended, gliding effortlessly over the razor-sharp, crystalline ice cliffs of an arctic landscape. The first light of sunrise paints the horizon in hues of fiery orange and soft pink, casting a warm glow that contrasts with the cold blue of the ice. The owl's fierce yellow eyes are focused, and tiny ice crystals are caught in its pristine white feathers, sparkling in the morning light. A sense of freedom, power, and raw natural beauty. Professional wildlife photography, shot on a high-speed camera, telephoto lens, 8K resolution, motion blur on the wingtips."
"Epic cosmic vista of a surreal, impossible staircase made of polished marble and glowing celestial energy, spiraling infinitely through the cosmos. The staircase twists and turns through vibrant, multi-colored nebulas, past distant, glittering galaxies and blazing supernovas. A lone, silhouetted figure is seen ascending the steps, heading towards a blindingly bright cosmic anomaly at the top. The scene is filled with a sense of wonder, mystery, and metaphysical journey. Surrealist digital painting, ultra-detailed, volumetric lighting, epic scale, trending on ArtStation, 8K."
)

for i in "${!instructions[@]}"; do
    echo "Generating image ${i}..."
    echo "Instruction: ${instructions[$i]}"
    
    model_path="OmniGen2/OmniGen2"
    python inference.py \
    --model_path "$model_path" \
    --num_inference_step 50 \
    --height 2048 \
    --width 1024 \
    --text_guidance_scale 3.5 \
    --instruction "${instructions[$i]}" \
    --output_image_path "outputs/output_t2i_$(($i + 12)).png" \
    --num_images_per_prompt 3
done
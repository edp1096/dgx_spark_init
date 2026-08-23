const source = 'https://app.sogni.ai/create?source=krea2-filter-bypass-2'

export const sogniPromptPresets = [
  {
    id: 'sogni-mixed-athlete', label: '운동선수 · 다중노출 콜라주', category: 'illustration', source,
    prompt: `A mixed-media collage features a female athlete leaping upward, leaving a stroboscopic trail of multiple limbs. Clad in a metallic silver bodysuit, she grasps glowing energy orbs in her sequentially trailing hands. Thick, sweeping impasto brushstrokes in vibrant pink, yellow, blue, green, and orange radiate downwards from her figure, creating an energetic wake of paint splatters. Set against a solid pale beige background, the low-angle composition blends photorealistic motion photography with expressive digital painting.`
  },
  {
    id: 'sogni-gallery-long-exposure', label: '미술관 · 장노출과 글자', category: 'photo', source,
    prompt: `A dynamic cinematic long-exposure fine art photograph inside a contemporary white-walled art gallery room "13B". A woman in a red dress viewed from behind stands completely still in front of a framed artwork. The framed piece on the gallery wall shows a woman laying on a grass painting, and below the frame on the wall label reads "KREA 2 Turbo." in larger font and "Woman on grass" in smaller font beneath it. Other gallery visitors captured in long exposure appear as semi-transparent ghostly blurred silhouettes drifting through the foreground. Contemplative atmosphere.`
  },
  {
    id: 'sogni-tile-portrait', label: '24×24 타일 · 뒤섞인 인물', category: 'graphic', source,
    prompt: `Analog collage portrait of a young woman with loose dark hair and a calm gaze, structured within a 24x24 grid of equal square tiles, most tiles correctly assembled to reveal her face, neck, and soft cream blouse against a muted terracotta background, but a deliberate scatter of roughly eight tiles are displaced—rotated ninety degrees or transposed to wrong positions across the grid—creating intentional visual ruptures where an eye lands near her shoulder and a section of background invades her jawline. Raw white tile borders accentuate each seam, warm directional studio lighting, fine-grain matte paper texture, mid-century editorial aesthetic, high-contrast graphic composition, the disorder itself reads as design.`
  },
  {
    id: 'sogni-coordinate-fashion', label: '좌표식 지시 · 패션 포즈', category: 'portrait', source,
    prompt: `1. Subject: A dynamic studio portrait of an athletic blonde woman in a beige knit crop top posing creatively.
2. Background: A seamless, completely solid minimalist white studio backdrop with no shadows or details.
3. Coordinates: Head and intense blue eyes of the model, tilted diagonally as she looks down the lens.
4. Coordinates: Both hands raised high above her head, tightly gripping and lifting her long blonde ponytail.
5. Coordinates: Long sleeves of a loose, textured beige open-knit cropped sweater framing her face.
6. Coordinates: A lean, highly defined athletic torso with prominent abdominal muscles in sharp focus.
7. Coordinates: Her right leg bent sharply at the knee, extending towards the lower left foreground.
8. Coordinates: String bikini bottoms in a matching muted beige hue tied at the hip.
9. Lighting: High-key studio lighting, intense softboxes from both sides erasing background shadows, crisp highlights on abdominal definition.
10. Style & Render: Professional fashion lookbook photography, ultra-sharp details, visible fabric weave texture, clean commercial color grading.`
  },
  {
    id: 'sogni-feather-angel', label: '프리즘 깃털 · 천사 인물', category: 'fantasy', source,
    prompt: `A Japanese woman angel stands centered with hair composed entirely of iridescent bird feathers catching the blinding brilliance of golden prismatic beams against an obsidian black backdrop. The radiant light fractures into rainbow-colored rays that slice across the scene, illuminating every delicate strand of downy plume like spun glass. Her form catches the high-contrast illumination while deep shadows swallow the edges of her frame. A cinematic 85mm lens captures this static pose with shallow depth of field, keeping the foreground feathers razor-sharp against the absolute black void behind her. The atmosphere hums with saturated neon violet and electric blue tones mixed with the stark white of the dazzling light source. Each feather reflects a spectrum of color like oil on water, shimmering under the intense focus of the camera lens.`
  },
  {
    id: 'sogni-animal-tower', label: '1960년대 서커스 · 동물 탑', category: 'animal', source,
    prompt: `A dynamic, low-angle photograph from 1960 capturing a surreal circus act: a tired giraffe with legs buckled under its own weight stands at the base of a precarious animal tower, upon whose back a hippopotamus stands, then a zebra, then a lion, and finally a relaxed lemur perched atop the lion's shoulders—all balancing precariously as if about to collapse. Two parrots fly swiftly past the camera in mid-air, their wings blurred by motion. The background is a packed circus arena filled with cheering spectators of all ages, many holding popcorn, glasses of Coca-Cola, and confetti. Bright theatrical circus lighting illuminates the animals from above and the front, casting sharp highlights and deep shadows that emphasize the tension and scale. The composition centers squarely on the stacked animals, framed tightly from below, with the crowd and flying parrots adding depth and energy.`
  },
  {
    id: 'sogni-synthwave-highway', label: '신스웨이브 · 네온 고속도로', category: 'fantasy', source,
    prompt: `A retro synthwave highway stretching toward a glowing, wireframe mountain range. The sky is a dark purple grid, dominated by a massive, low-hanging neon pink sun with horizontal segment lines. The highway is a dark, reflective surface lined with neon magenta light strips that stretch into the horizon. A sleek, futuristic 1980s sports car is seen from behind, its red taillights leaving a faint trail of light. The overall aesthetic is clean, geometric, and vibrant.`
  },
  {
    id: 'sogni-double-exposure', label: '인물 실루엣 · 들판 이중노출', category: 'portrait', source,
    prompt: `A surreal double-exposure portrait of a woman seen from behind, her head and bare shoulders forming a dark silhouette against a muted misty sky. Her hair is loosely tied in a messy bun with fine stray strands. Inside the silhouette is a vast golden field at sunset, with tall dry grass, distant dark mountains, a glowing low sun, and a lone small figure standing in the center facing the horizon. Above her head, black birds fly in scattered motion, their silhouettes fading into the pale sky. Dreamlike conceptual realism, vertical composition, centered subject, soft film grain, subtle canvas texture, warm amber light inside the body, cool gray-beige background, quiet melancholy mood, symbolic atmosphere, painterly photoreal detail, elegant contrast between inner landscape and outer shadow, balanced gallery-poster composition.`
  },
  {
    id: 'sogni-mouse-library', label: '생쥐 도서관 · 단면 디오라마', category: 'animal', source,
    prompt: `A magical nighttime cross-section diorama of a mouse's private library inside a hollow willow tree. A tiny white mouse in a navy hand-knit sweater reads quietly in a plush forest-green velvet armchair, illuminated by several glowing fireflies floating inside delicate glass jars. Curving wooden bookshelves are filled with matchbook-sized volumes, tiny botanical journals, seed packets, dried herbs, and miniature climbing vines. A small ladder leads to an upper reading loft beneath roots and twisting branches. Deep midnight blue exterior, warm candlelit interior, enchanting handcrafted miniature realism, cinematic soft focus.`
  },
  {
    id: 'sogni-alf-vintage', label: '복고풍 ALF · 코믹 공포', category: 'photo', source,
    prompt: `A humorous, vintage-style photograph of the iconic alien character ALF standing indoors in front of white paneled doors. ALF is dressed in a comical human outfit: a long blue patterned dress with a paisley or swirl design and short puffed sleeves, topped with a wide-brimmed light-pink sun hat. He is holding a silver kitchen knife upright in his right hand, posing as if threatening or ready to strike. His expression is serious and focused. The lighting is flat and typical of indoor home photography from the late 1980s or early 1990s. The image has a slightly grainy, low-resolution quality, enhancing its meme-like retro aesthetic.`
  },
  {
    id: 'sogni-wide-angle-portrait', label: '과장 광각 · 로우 앵글 인물', category: 'portrait', source,
    prompt: `A close-up of the face of a young European woman with a curly brown afro and large white fashionable sunglasses reflecting a Los Angeles beach. She has a nose piercing and is drinking coffee from a cup with a straw. She is looking up. The composition focuses on her face from below, with an extremely exaggerated wide camera angle, exaggerated shapes, and a dynamic composition.`
  },
  {
    id: 'sogni-cat-typography', label: '고양이 털 · CAT 글자', category: 'animal', source,
    prompt: `Portrait of a black-and-white cat. A spot of white fur forms the diagonal text "CAT", with each character rendered at a different size.`
  },
  {
    id: 'sogni-korean-techwear', label: '한국인 테크웨어 · 로우 앵글', category: 'portrait', source,
    prompt: `Low-angle full-body fashion photograph of a Korean woman with a chunky black bob haircut, bright blue tips, short blunt bangs, a broad Korean nose, big round head and cheeks, and hoop earrings. She is seated on a white box against a clear blue sky, wearing a black Nike-symbol hoodie, oversized Japanese tech-wear sweatpants, and Air Jordan Mid 1 shoes. Sunlight illuminates her face from the front while she gazes neutrally out of frame, centered with balanced framing. Soft natural lighting enhances textures and contours, with no additional props or characters and a clean minimalist background emphasizing the subject and outfit details.`
  },
  {
    id: 'sogni-blue-hour-crosswalk', label: '블루아워 · 횡단보도 스냅', category: 'photo', source,
    prompt: `A woman in her mid-20s with a short curly afro, wearing a bright yellow midi dress and white sneakers, crossing a wide urban crosswalk during blue hour. Pose: mid-step, dress flaring slightly, looking over her shoulder with a spontaneous joyful laugh. Style: documentary street portrait. Lighting: soft ambient city light mixed with cool twilight from above and warm storefront glow hitting her side. Composition: foreground pedestrian blurred, subject in sharp focus in the mid-ground, background traffic lights and bodega signs creating layered depth. 35mm lens with shallow depth of field and slight motion blur in the legs.`
  },
  {
    id: 'sogni-origami-goose', label: '거위 · 일본 종이접기', category: 'animal', source,
    prompt: `A goose rendered in an origami art style using the traditional Japanese paper-folding technique, forming an intricate sculptural design from folded paper.`
  },
  {
    id: 'sogni-pixel-face', label: '픽셀 붕괴 · 흑백 남성', category: 'graphic', source,
    prompt: `Film grain, low contrast, long shadows, cinematic mood, black-and-white photography. A man in profile has his face completely masked by a dense fluid mass of swirling black-and-grey pixels and data-like horizontal lines. He has a slender build and dark wavy hair combed back, with only his ear and jawline visible beneath the distortion. He wears a dark tailored suit over a crisp white shirt and subtly patterned tie, holding a smartphone toward the pixelated void in a pose merging human form with digital abstraction. A stark textured urban background with concrete pillars and a barred window, diffused dramatic light, soft defined shadows, raw analog-film aesthetic, unsettling and introspective mood.`
  },
  {
    id: 'sogni-glitch-cat', label: '고양이 · 디지털 글리치', category: 'animal', source,
    prompt: `A glitched photo of a cat with digital glitch distortions, fragmented and corrupted pixels creating visual artifacts. The cat's form is partially disintegrated, with parts of its body displaced and duplicated across the frame. Vibrant neon colors bleed through static noise and data-corruption effects. The cat sits against a glitched white background with broken scan lines and chromatic aberration, creating a warped-reality aesthetic.`
  },
  {
    id: 'sogni-uyuni-meteor', label: '우유니 소금사막 · 별똥별', category: 'photo', source,
    prompt: `A twenty-something Japanese woman with short brown hair gazes upward under cool starlight beside an elegant bicycle, her silhouette framed by a one-piece dress with a structured bodice and flowing skirt. The vast Uyuni Salt Lake stretches behind her like an endless mirror, capturing every celestial point in vivid reflection. Long-exposure light shifts from deep indigo to bruised violet across the night sky while a single giant meteor leaves a ghostly trail of white fire. Rendered as a grainy 35mm film photograph with shallow depth of field and soft bokeh emphasizing the atmospheric glow and texture of the water.`
  },
  {
    id: 'sogni-overhead-ballerina', label: '발레리나 · 정수리 포어쇼트닝', category: 'photo', source,
    prompt: `A ballerina caught mid-leap, photographed from directly above. Extreme overhead foreshortening compresses the full body into an abstract geometric shape—arms wide, tutu a white halo, pointed feet impossibly small below. The camera is tilted at a 22-degree Dutch angle. A polished black marley floor acts as a mirror beneath her. High-key studio lighting bleaches the tutu to pure white against deep charcoal, with a long shadow cutting diagonally across the composition, architectural minimalism, severe fashion-editorial mood, absolute stillness in motion.`
  },
  {
    id: 'sogni-iphone-car', label: '자동차 안 · 스마트폰 스냅', category: 'portrait', source,
    prompt: `Shot on an iPhone inside a parked car, with bright midday sunlight through the windows and a casual social-media photo style. A young woman reclines in a red leather passenger seat with one knee raised toward the camera and the other leg folded across the seat. She wears a loose white V-neck T-shirt, light blue jeans, layered necklaces, bracelets, and a ring. One hand holds a lit rolled blunt near the window while smoke drifts across her face and curls around the car roof. Her expression is hazy and relaxed, with lowered eyelids and slightly parted lips. The background shows the car door, window glare, outdoor buildings and trees, red interior panels, and strong sun reflections. Authentic smartphone photo, natural harsh daylight, soft smoke haze, candid car-seat composition.`
  },
  {
    id: 'sogni-branch-headphones', label: '나뭇가지 · 조형적 헤드폰', category: '3d', source,
    prompt: `Product design of sculptural headphones made from a single piece of raw natural branch wood with bark, extending upward into irregular branching limbs. The branches seamlessly support the ear cups, creating an organic asymmetrical silhouette against a pure white seamless background. Ultra-realistic high-quality textures and soft studio lighting.`
  },
  {
    id: 'sogni-cat-bento', label: '고양이 도시락 · 플랫레이', category: 'photo', source,
    prompt: `A perfectly arranged Japanese bento box featuring a sleeping cat made of fluffy white rice, with nori details for its eyes and whiskers, snuggled under a blanket made of a thin egg omelette. Surrounding the rice cat are colorful cherry tomatoes and broccoli florets. Bright flat-lay food photography.`
  }
]

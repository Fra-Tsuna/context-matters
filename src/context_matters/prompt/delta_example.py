from numpy import array

DOMAIN_DATA_EXAMPLE = {
    "scene": ["kemblesville", "parole"],
    "add_obj": None,
    "add_act": [
        "mop_floor(<agent>, <mop>, <room>): For mopping, <mop> must be a mop and clean, <agent> must be holding the <mop>, <agent> and <mop> should be in <room>, floor is dirty. As result, floor will be clean and mop will be dirty."
    ],
    "goal": "Clean the floors.",
    "gt_cost": {
        "kemblesville": 27,
        "parole": 0  # FIXME
    },
    "item_keep": ["mop_11", "bucket_13"],
    "subgoal": [
        "Clean floor in kitchen_2",
        "Clean floor in living_room_1"
    ],
    "subgoal_pddl": [
        """
    (:goal
        (and
            (clean-floor kitchen_2)
        )
    )\n""",
        """
    (:goal
        (and
            (clean-floor living_room_1)
        )
    )\n"""
    ],
    "env_state": [
        "item_is_mop(<item>): <item> is a mop.",
        "item_is_bucket(<item>): <item> is a bucket.",
        "item_is_glass(<item>): <item> is a glass.",
        "floor_clean(<room>): <room> has a clean floor.",
    ]
}

DOMAIN_EXAMPLE = """
;Header and description
(define (domain laundry)

    (:requirements :strips :typing :adl)

    ; Begin types
    (:types
        agent room item
    )
    ; End types

    ; Begin predicates
    (:predicates
        (neighbor ?r1 - room ?r2 - room)
        (agent_at ?a - agent ?r - room)
        (item_at ?i - item ?r - room)
        (item_pickable ?i - item)
        (item_turnable ?i - item)
        (agent_loaded ?a - agent)
        (agent_has_item ?a - agent ?i - item)
        (item_is_cloth ?i - item)
        (item_is_detergent ?i - item)
        (item_is_wash_machine ?i - item)
        (cloth_clean ?i - item)
    )
    ; End predicates

    ; Begin actions
    (:action move_to
        :parameters (?a - agent ?r1 - room ?r2 - room)
        :precondition (and
            (agent_at ?a ?r1)
            (neighbor ?r1 ?r2)
        )
        :effect (and
            (not(agent_at ?a ?r1))
            (agent_at ?a ?r2)
        )
    )

    (:action pick
        :parameters (?a - agent ?i - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (item_at ?i ?r)
            (item_pickable ?i)
            (not(agent_loaded ?a))
            (not(agent_has_item ?a ?i))
        )
        :effect (and
            (agent_at ?a ?r)
            (not(item_at ?i ?r))
            (agent_loaded ?a)
            (agent_has_item ?a ?i)
        )
    )

    (:action drop
        :parameters (?a - agent ?i - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (not(item_at ?i ?r))
            (item_pickable ?i)
            (agent_loaded ?a)
            (agent_has_item ?a ?i)
        )
        :effect (and
            (agent_at ?a ?r)
            (item_at ?i ?r)
            (not(agent_loaded ?a))
            (not(agent_has_item ?a ?i))
        )
    )

    (:action launder
        :parameters (?a - agent ?i1 - item ?i2 - item ?i3 - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (item_at ?i1 ?r)
            (item_at ?i2 ?r)
            (item_at ?i3 ?r)
            (item_is_cloth ?i1)
            (item_is_detergent ?i2)
            (item_is_wash_machine ?i3)
            (item_pickable ?i1)
            (item_pickable ?i2)
            (item_turnable ?i3)
            (not(cloth_clean ?i1))
            (not(agent_loaded ?a))
            (not(agent_has_item ?a ?i1))
        )
        :effect (and
            (cloth_clean ?i1)
        )
    )
    ; End actions

)
"""

DOMAIN_EXAMPLE_OURS = """
(define (domain house-cleaning-domain)

    (:requirements
        :strips
        :typing
    )

    ; Begin types
    (:types
        room locatable - object
        robot grabbable bin - locatable
        disposable mop - grabbable

    )
    ; End types

    ; Begin predicates
    (:predicates
        (at ?something - locatable ?where - room)
        (is-holding ?who - robot ?something - grabbable)
        (is-free ?who - robot)
        (thrashed ?what - disposable)
        (is-clean ?what - mop)
        (is-dirty ?what - mop)
        (dirty-floor ?what - room)
        (clean-floor ?what - room)
    )
    ; End predicates

    ; Begin actions
    (:action move_to
        :parameters (?who - robot ?from - room ?to - room)
        :precondition (and (at ?who ?from))
        :effect (and (not (at ?who ?from)) (at ?who ?to))
    )
    
    (:action grab
        :parameters (?who - robot ?what - grabbable ?where - room)
        :precondition (and (at ?who ?where) (at ?what ?where) (is-free ?who))
        :effect (and (not (at ?what ?where)) (is-holding ?who ?what) (not (is-free ?who)))
    )
    
    (:action drop
        :parameters (?who - robot ?what - grabbable ?where - room)
        :precondition (and (at ?who ?where) (is-holding ?who ?what))
        :effect (and (not (is-holding ?who ?what)) (is-free ?who) (at ?what ?where))
    )
    
    (:action throw_away
        :parameters (?who - robot ?what - disposable ?in - bin ?where - room)
        :precondition (and (at ?who ?where) (is-holding ?who ?what) (at ?in ?where))
        :effect (and (not (is-holding ?who ?what)) (is-free ?who) (thrashed ?what) (not (at ?what ?where)))
    )
    
    (:action mop_floor
        :parameters (?who - robot ?with - mop ?where - room)
        :precondition (and (at ?who ?where) (is-holding ?who ?with)
                    (is-clean ?with) (dirty-floor ?where))
        :effect (and (not (dirty-floor ?where)) (not (is-clean ?with))
                (clean-floor ?where) (is-dirty ?with)    
        )
    )
    
    (:action clean_mop
        :parameters (?who - robot ?what - mop)
        :precondition (and (is-holding ?who ?what) (is-dirty ?what))
        :effect (and (not (is-dirty ?what)) (is-clean ?what))
    )
    ; End actions
)
"""

#SCENE_EXAMPLE = example.get_scenes(DOMAIN_EXAMPLE)[0]
SCENE_EXAMPLE_OURS = \
{
    'room': {
        1: {
            'floor_area': 7.309859732351093,
            'floor_number': 'A',
            'id': 1,
            'location': array([-0.7197105, -1.610225 ,  1.1466205]),
            'scene_category': 'dining_room',
            'size': array([2.5, 1.3, 0.8]),
            'volume': 7.856456590462219,
            'parent_building': 70
        },
        2: {
            'floor_area': 7.309859732351093,
            'floor_number': 'B',
            'id': 2,
            'location': array([-1.7197105  , -2.610225   ,  1.255566205]),
            'scene_category': 'kitchen',
            'size': array([1.836419, 1.91739 , 2.326879]),
            'volume': 5.21398719237,
            'parent_building': 70
        },
        3: {
            'floor_area': 7.309859732351093,
            'floor_number': 'A',
            'id': 3,
            'location': array([-0.8297505 , -1.717225  ,  1.77466205]),
            'scene_category': 'bathroom',
            'size': array([1.836419, 1.91739 , 2.326879]),
            'volume': 4.948372943,
            'parent_building': 70
        }
    },
    'object': {
        11: {
            'action_affordance': ['wash', 'clean', 'clean with'],
            'floor_area': 1.7741857552886344,
            'surface_coverage': 1.0771061806828595,
            'class_': 'mop',
            'id': 11,
            'location': array([-0.77102688, -1.56634777,  0.45304914]),
            'material': None,
            'tactile_texture': None,
            'visual_texture': None,
            'volume': 0.13050794184989123,
            'parent_room': 2,
            'description': 'A cleaning mop for mopping floors.'
        },
        15: {
            'action_affordance': ['wash', 'clean', 'break', 'move'],
            'floor_area': 1.7741857552886344,
            'surface_coverage': 0.02222,
            'class_': 'glass',
            'id': 15,
            'material': 'glass',
            'tactile_texture': None,
            'visual_texture': None,
            'volume': 0.13050794184989123,
            'parent_room': 1,
            'description': 'A drinking glass.'
        },
        13: {
            'action_affordance': ['wash', 'clean', 'clean with'],
            'floor_area': 1.7741857552886344,
            'surface_coverage': 0.02222,
            'class_': 'bucket',
            'id': 13,
            'material': 'ceramic',
            'tactile_texture': None,
            'visual_texture': None,
            'volume': 0.13050794184989123,
            'parent_room': 3,
            'description': 'A ceramic bucket for cleaning.'
        }
    }
}


# NOTICE: taken from DELTA/data/pddl/kemblesville_laundry_problem.pddl
PROBLEM_EXAMPLE = """
(define (problem kemblesville)
    (:domain laundry)

    ; Begin objects
    (:objects
        robot - agent
        bathroom closet_1 closet_2 corridor_1 corridor_2 bedroom_1 bedroom_2 living_room kitchen - room
        detergent clothes wash_machine - item
    )
    ; End objects

    ; Begin init
    (:init
        ; Connections
        (neighbor bathroom corridor_1)
        (neighbor bathroom closet_1)
        (neighbor closet_1 bathroom)
        (neighbor closet_1 bedroom_1)
        (neighbor closet_2 corridor_2)
        (neighbor closet_2 kitchen)
        (neighbor corridor_1 bedroom_1)
        (neighbor corridor_1 bedroom_2)
        (neighbor corridor_1 bathroom)
        (neighbor corridor_1 living_room)
        (neighbor corridor_2 closet_2)
        (neighbor corridor_2 kitchen)
        (neighbor corridor_2 living_room)
        (neighbor bedroom_1 corridor_1)
        (neighbor bedroom_1 closet_1)
        (neighbor bedroom_2 corridor_1)
        (neighbor living_room corridor_1)
        (neighbor living_room corridor_2)
        (neighbor kitchen corridor_2)
        (neighbor kitchen closet_2)

        ; Positions
        (agent_at robot living_room)
        (item_at detergent bathroom)
        (item_at clothes closet_1)
        (item_at wash_machine kitchen)

        ; Attributes
        (item_is_cloth clothes)
        (item_is_detergent detergent)
        (item_is_wash_machine wash_machine)
        (item_pickable clothes)
        (item_turnable wash_machine)
        (item_pickable detergent)
        (not(cloth_clean clothes))
    )
    ; End init

    ; Begin goal
    (:goal
        (and
            (cloth_clean clothes)
            (item_at clothes bedroom_1)
        )
    )
    ; End goal
)
"""

PROBLEM_EXAMPLE_OURS = """
(define (problem house_cleaning)
    (:domain house-cleaning-domain)

    ; Begin objects
    (:objects
        robot - robot
        kitchen_2 living_room_1 - room
        mop_11 - mop
        bucket_13 - grabbable
    )
    ; End objects

    ; Begin init
    (:init
        (at robot kitchen_2)
        (at mop_11 kitchen_2)
        (at bucket_13 kitchen_2)
        (is-free robot)
        (is-clean mop_11)
        (dirty-floor kitchen_2)
        (dirty-floor living_room_1)
    )
    ; End init

    ; Begin goal
    (:goal
        (and
            (clean-floor kitchen_2)
            (clean-floor living_room_1)
        )
    )
    ; End goal
)
"""

HARDCODED_DOMAIN_EXAMPLE_NAME = "house-cleaning-domain"
def load_example_data(domain_name=None):
    return (DOMAIN_EXAMPLE_OURS, SCENE_EXAMPLE_OURS, PROBLEM_EXAMPLE_OURS, DOMAIN_DATA_EXAMPLE)
